from pathlib import Path
import io
import json
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances


st.set_page_config(
    page_title="Cryo-EM particle explorer",
    page_icon="🔬",
    layout="wide",
)

st.title("Cryo-EM particle explorer")

st.caption(
    "Explore particle representations, clustering and representative images. "
)


left_col, title_col, right_col = st.columns(
    [1, 2, 1],
    vertical_alignment="center"
)

video_path = "movies/dimer_small.webm"
video_path_1 = "movies/tetramer_small.webm"

with open(video_path, "rb") as f:
    video_bytes = f.read()
with open(video_path_1, "rb") as f:
    video_bytes_1 = f.read()
    
video_base64 = base64.b64encode(video_bytes).decode("utf-8")
video_base64_1 = base64.b64encode(video_bytes_1).decode("utf-8")

video_html = f"""
<video width="100%" autoplay loop muted playsinline
    style="background: transparent;">
    <source src="data:video/mp4;base64,{video_base64}" type="video/webm">
    Your browser does not support this video.
</video>
"""

video_html_1 = f"""
<video width="100%" autoplay loop muted playsinline
    style="background: transparent;">
    <source src="data:video/mp4;base64,{video_base64_1}" type="video/webm">
    Your browser does not support this video.
</video>
"""

with left_col:
    components.html(video_html, height=280)

with title_col:
    st.markdown(
        "<h1 style='text-align:center;'>Is it a dimer or a tetramer?</h1>",
        unsafe_allow_html=True
    )

with right_col:
    components.html(video_html_1, height=280)


st.write(
    "Here, we want to differentiate between dimer and tetramer of a protein directly from cryo-EM images. Below is our workflow:"
)

st.image("images/workflow_ml4ls.png")

# ---------------------------------------------------------
# Image utilities
# ---------------------------------------------------------

def image_from_array(array):
    array = np.asarray(array)

    if array.ndim == 1:
        side = int(np.sqrt(array.size))
        if side * side != array.size:
            return None
        array = array.reshape(side, side)

    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)

    if array.ndim == 3 and array.shape[-1] not in (1, 3, 4):
        array = array[0]

    array = np.squeeze(array)

    if array.ndim != 2 and not (
        array.ndim == 3 and array.shape[-1] in (1, 3, 4)
    ):
        return None

    array = array.astype(float)

    if not np.isfinite(array).any():
        return None

    low, high = np.nanpercentile(array, [1, 99])

    if high <= low:
        return None

    array = np.clip((array - low) / (high - low), 0, 1)

    return Image.fromarray((array * 255).astype(np.uint8))


def read_image_zip(upload):
    if upload is None:
        return {}

    images = {}

    with zipfile.ZipFile(upload) as archive:
        for filename in archive.namelist():

            if not filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
            ):
                continue

            try:
                image = Image.open(
                    io.BytesIO(archive.read(filename))
                ).convert("L")

                images[Path(filename).name] = image.copy()
                images[Path(filename).stem] = image.copy()

            except Exception:
                continue

    return images


def find_image(images, name):
    if not images:
        return None

    key = str(name)

    return (
        images.get(key)
        or images.get(Path(key).name)
        or images.get(Path(key).stem)
    )


# ---------------------------------------------------------
# Feature analysis utilities
# ---------------------------------------------------------

def numeric_matrix(dataframe, excluded):
    columns = [
        column
        for column in dataframe.select_dtypes(include=np.number).columns
        if column not in excluded
    ]

    if not columns:
        raise ValueError(
            "No numeric feature columns found. "
            "Your CSV needs numeric feature columns."
        )

    matrix = dataframe[columns].replace(
        [np.inf, -np.inf], np.nan
    )

    matrix = matrix.fillna(matrix.median()).fillna(0)

    matrix = matrix.to_numpy(dtype=float)

    if matrix.shape[0] < 3:
        raise ValueError("At least three particles are required.")

    return matrix, columns


def reduce_space(matrix, method, seed, neighbors, min_dist):
    scaled = StandardScaler().fit_transform(matrix)

    n_components = min(
        3,
        scaled.shape[0] - 1,
        scaled.shape[1],
    )

    if method == "PCA":
        model = PCA(
            n_components=n_components,
            random_state=seed,
        )

        embedding = model.fit_transform(scaled)

        variance = model.explained_variance_ratio_[:2].sum()

        return (
            embedding,
            f"PCA (PC1 + PC2 explain {variance:.1%} of variance)",
        )

    if method == "ICA":
        model = FastICA(
            n_components=n_components,
            random_state=seed,
            max_iter=1000,
            tol=0.001,
        )

        embedding = model.fit_transform(scaled)

        return embedding, "ICA independent components"

    try:
        import umap
    except ImportError as error:
        raise RuntimeError(
            "UMAP is not installed. Add umap-learn to requirements.txt."
        ) from error

    model = umap.UMAP(
        n_components=2,
        n_neighbors=min(neighbors, len(scaled) - 1),
        min_dist=min_dist,
        random_state=seed,
    )

    embedding = model.fit_transform(scaled)

    return embedding, "UMAP"


def cluster_data(matrix, method, k, eps, min_samples, min_cluster):
    if method == "K-means":
        return KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42,
        ).fit_predict(matrix)

    if method == "DBSCAN":
        return DBSCAN(
            eps=eps,
            min_samples=min_samples,
        ).fit_predict(matrix)

    try:
        import hdbscan
    except ImportError as error:
        raise RuntimeError(
            "HDBSCAN is not installed. Add hdbscan to requirements.txt."
        ) from error

    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster,
        min_samples=min_samples,
    ).fit_predict(matrix)


def representative_index(matrix, indices):
    subset = matrix[indices]
    center = subset.mean(axis=0, keepdims=True)

    distances = pairwise_distances(subset, center).ravel()

    return int(indices[np.argmin(distances)])


def plot_embedding(embedding, labels, particle_ids, title):
    data = pd.DataFrame({
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "cluster": labels.astype(str),
        "particle": particle_ids,
    })

    if embedding.shape[1] > 2:
        data["z"] = embedding[:, 2]

        figure = px.scatter_3d(
            data,
            x="x",
            y="y",
            z="z",
            color="cluster",
            hover_name="particle",
            title=title,
        )
    else:
        figure = px.scatter(
            data,
            x="x",
            y="y",
            color="cluster",
            hover_name="particle",
            title=title,
        )

    figure.update_layout(height=650)

    st.plotly_chart(figure, use_container_width=True)


# ---------------------------------------------------------
# Saved-results utilities
# ---------------------------------------------------------

def find_outputs_folder():
    """Find outputs in the current directory or one of its parents."""

    for path in (Path.cwd(), *Path.cwd().parents):
        candidate = path / "outputs"

        if candidate.is_dir():
            return candidate

    return None


def resolve_checkpoint_path(outputs_folder, saved_path):
    """
    Resolve a checkpoint path.

    Supports both relative paths and old absolute paths from
    the computer where the notebook was originally run.
    """

    path = Path(saved_path)

    if path.exists():
        return path

    fallback = outputs_folder / path.name

    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not find checkpoint folder: {saved_path}"
    )


def load_saved_run(outputs_folder, run_name):
    """Load a saved clustering run without restoring the original MRC data."""

    index_file = outputs_folder / "clustering_index.json"

    with open(index_file, encoding="utf-8") as file:
        entries = json.load(file)

    if run_name not in entries:
        raise KeyError(f"Saved run not found: {run_name}")

    checkpoint = resolve_checkpoint_path(
        outputs_folder,
        entries[run_name],
    )

    settings_file = checkpoint / "settings.json"
    table_file = checkpoint / "table.json"
    arrays_file = checkpoint / "arrays.npz"

    if not settings_file.exists():
        raise FileNotFoundError("Missing settings.json in checkpoint.")

    if not table_file.exists():
        raise FileNotFoundError("Missing table.json in checkpoint.")

    if not arrays_file.exists():
        raise FileNotFoundError("Missing arrays.npz in checkpoint.")

    with open(settings_file, encoding="utf-8") as file:
        settings = json.load(file)

    table = pd.read_json(table_file, orient="table")

    with np.load(arrays_file, allow_pickle=False) as archive:
        arrays = {
            key: archive[key].copy()
            for key in archive.files
        }

    return {
        "checkpoint_directory": checkpoint,
        "settings": settings,
        "table": table,
        "arrays": arrays,
    }


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Data")

    mode = st.radio(
        "Input source",
        [
            "Feature CSV (flexible)",
            "Saved notebook clustering results",
        ],
    )

    if mode == "Feature CSV (flexible)":
        image_zip = st.file_uploader(
            "Optional particle image archive (.zip)",
            type=["zip"],
        )

        image_map = read_image_zip(image_zip)

        st.caption(
            "Image archive filenames should match particle IDs in the table."
        )


# =========================================================
# MODE 1: FEATURE CSV
# =========================================================

if mode == "Feature CSV (flexible)":

    csv_file = st.file_uploader(
        "Upload particle feature table (.csv)",
        type=["csv"],
    )

    if csv_file is None:
        st.info(
            "Upload a CSV with one row per particle, an ID column "
            "and numeric feature columns. You can optionally include "
            "a precomputed cluster column."
        )
        st.stop()

    dataframe = pd.read_csv(csv_file)

    id_candidates = [
        column
        for column in (
            "particle_id",
            "image_name",
            "image_id",
            "id",
        )
        if column in dataframe.columns
    ]

    id_col = st.selectbox(
        "Particle ID column",
        id_candidates or list(dataframe.columns),
    )

    particle_ids = dataframe[id_col].astype(str).to_numpy()

    existing_cluster = next(
        (
            column
            for column in ("cluster", "cluster_id", "labels")
            if column in dataframe.columns
        ),
        None,
    )

    excluded = {id_col}

    if existing_cluster:
        excluded.add(existing_cluster)

    if "known_class" in dataframe.columns:
        excluded.add("known_class")

    try:
        matrix, feature_columns = numeric_matrix(
            dataframe,
            excluded,
        )
    except Exception as error:
        st.error(str(error))
        st.stop()

    # Analysis controls
    with st.sidebar:
        st.header("Analysis")

        reducer = st.selectbox(
            "Embedding",
            ["PCA", "ICA", "UMAP"],
        )

        cluster_method = st.selectbox(
            "Clustering",
            ["K-means", "DBSCAN", "HDBSCAN"],
        )

        cluster_space = st.selectbox(
            "Cluster on",
            ["Scaled features", "2D/3D embedding"],
        )

        k = st.slider(
            "K-means clusters",
            2,
            min(20, max(2, len(dataframe) - 1)),
            min(4, max(2, len(dataframe) - 1)),
        )

        eps = st.slider(
            "DBSCAN eps",
            0.05,
            5.0,
            0.7,
            0.05,
        )

        min_samples = st.slider(
            "Minimum samples",
            2,
            30,
            5,
        )

        min_cluster = st.slider(
            "HDBSCAN minimum cluster size",
            2,
            min(50, max(2, len(dataframe))),
            min(10, max(2, len(dataframe))),
        )

        seed = st.number_input(
            "Random seed",
            0,
            100000,
            42,
        )

        neighbors = st.slider(
            "UMAP neighbors",
            2,
            min(50, max(2, len(dataframe) - 1)),
            min(15, max(2, len(dataframe) - 1)),
        )

        min_dist = st.slider(
            "UMAP min_dist",
            0.0,
            0.99,
            0.1,
        )

    try:
        embedding, embedding_label = reduce_space(
            matrix,
            reducer,
            int(seed),
            neighbors,
            min_dist,
        )

        if cluster_space == "Scaled features":
            cluster_input = StandardScaler().fit_transform(matrix)
        else:
            cluster_input = embedding

        if existing_cluster:
            labels = (
                dataframe[existing_cluster]
                .fillna(-1)
                .astype(int)
                .to_numpy()
            )
        else:
            labels = cluster_data(
                cluster_input,
                cluster_method,
                k,
                eps,
                min_samples,
                min_cluster,
            )

    except Exception as error:
        st.error(str(error))
        st.stop()

    output = dataframe.copy()
    output["app_cluster"] = labels

    # Latent space
    st.subheader("Latent space")

    plot_embedding(
        embedding,
        labels,
        particle_ids,
        embedding_label + " · " + (
            existing_cluster or cluster_method
        ),
    )

    # Summary metrics
    counts = pd.Series(labels).value_counts()

    non_noise_clusters = [
        label for label in counts.index if label != -1
    ]

    a, b, c = st.columns(3)

    a.metric("Particles", len(dataframe))
    b.metric("Clusters (excluding noise)", len(non_noise_clusters))
    c.metric("Noise points", int(counts.get(-1, 0)))

    # Representatives
    st.subheader("Cluster representatives")

    representatives = {}

    unique_labels = sorted(np.unique(labels))

    columns = st.columns(min(4, max(1, len(unique_labels))))

    for index, label in enumerate(unique_labels):

        indices = np.flatnonzero(labels == label)

        representative = representative_index(
            matrix,
            indices,
        )

        representatives[label] = representative

        with columns[index % len(columns)]:
            st.markdown(
                f"**Cluster {label}** · {len(indices)} particles"
            )

            image = find_image(
                image_map,
                particle_ids[representative],
            )

            if image is not None:
                st.image(
                    image,
                    caption=f"Representative: {particle_ids[representative]}",
                    use_container_width=True,
                )
            else:
                st.caption(
                    f"Representative particle: {particle_ids[representative]} "
                    "(no matching image in ZIP)"
                )

    # Inspect cluster
    st.subheader("Inspect a cluster")

    chosen_cluster = st.selectbox(
        "Cluster",
        unique_labels,
        format_func=lambda label: (
            f"Cluster {label} ({int((labels == label).sum())} particles)"
        ),
    )

    members = np.flatnonzero(labels == chosen_cluster)

    st.dataframe(
        output.iloc[members].head(500),
        use_container_width=True,
    )

    st.download_button(
        "Download assignments CSV",
        output.to_csv(index=False).encode(),
        "particle_cluster_assignments.csv",
        "text/csv",
    )

    # Find particle
    st.subheader("Find a particle")

    particle = st.selectbox(
        "Existing particle",
        particle_ids.tolist(),
    )

    row = int(np.flatnonzero(particle_ids == particle)[0])
    label = labels[row]
    representative = representatives[label]

    left, right = st.columns(2)

    with left:
        st.markdown(f"**Particle {particle} · cluster {label}**")

        image = find_image(image_map, particle)

        if image is not None:
            st.image(image, use_container_width=True)
        else:
            st.info("No image matched this particle ID in the ZIP.")

    with right:
        st.markdown(f"**Representative of cluster {label}**")

        image = find_image(
            image_map,
            particle_ids[representative],
        )

        if image is not None:
            st.image(image, use_container_width=True)
        else:
            st.info(
                f"Representative ID: {particle_ids[representative]}"
            )

    # New particle upload
    st.subheader("Upload a new particle")

    new_image_file = st.file_uploader(
        "New particle image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="new_particle",
    )

    if new_image_file is not None:

        try:
            from app_feature_pipeline import extract_features

            new_image = Image.open(new_image_file).convert("L")

            new_features = np.asarray(
                extract_features(new_image),
                dtype=float,
            ).reshape(1, -1)

            if new_features.shape[1] != matrix.shape[1]:

                st.error(
                    f"Feature pipeline returned {new_features.shape[1]} "
                    f"values; expected {matrix.shape[1]}."
                )

            else:
                scaler = StandardScaler().fit(matrix)

                distances = pairwise_distances(
                    scaler.transform(new_features),
                    scaler.transform(matrix),
                ).ravel()

                nearest = int(np.argmin(distances))
                assigned_cluster = labels[nearest]

                st.success(
                    f"Nearest-reference assignment: cluster {assigned_cluster}"
                )

                st.caption(
                    f"Nearest particle: {particle_ids[nearest]} · "
                    f"standardized feature distance: {distances[nearest]:.4f}. "
                    "This is a nearest-neighbour lookup, not a calibrated probability."
                )

                left, right = st.columns(2)

                with left:
                    st.image(
                        new_image,
                        caption="Uploaded particle",
                        use_container_width=True,
                    )

                with right:
                    st.markdown(
                        f"**Representative of cluster {assigned_cluster}**"
                    )

                    representative = representatives[assigned_cluster]

                    image = find_image(
                        image_map,
                        particle_ids[representative],
                    )

                    if image is not None:
                        st.image(image, use_container_width=True)
                    else:
                        st.write(
                            f"Representative ID: {particle_ids[representative]}"
                        )

        except ModuleNotFoundError:
            st.warning(
                "New-image assignment is not enabled yet. "
                "Add app_feature_pipeline.py with extract_features(image), "
                "using exactly the same preprocessing and feature order as "
                "the CSV. Existing-particle lookup works now."
            )

        except Exception as error:
            st.error(f"Could not process uploaded image: {error}")


# =========================================================
# MODE 2: SAVED NOTEBOOK CLUSTERING RESULTS
# =========================================================

else:

    st.subheader("Saved notebook clustering results")

    st.write(
        "This mode loads the clustering results already saved by your "
        "notebook. You do not need to upload a CSV or rerun clustering."
    )

    outputs_folder = find_outputs_folder()

    if outputs_folder is None:
        st.warning(
            "Could not find an outputs folder. Make sure the outputs "
            "folder is included in your project."
        )
        st.stop()

    index_file = outputs_folder / "clustering_index.json"

    if not index_file.exists():
        st.warning(
            "No clustering_index.json found in outputs. "
            "Save at least one named clustering result in the notebook first."
        )
        st.stop()

    try:
        with open(index_file, encoding="utf-8") as file:
            entries = json.load(file)

    except Exception as error:
        st.error(f"Could not read clustering_index.json: {error}")
        st.stop()

    if not entries:
        st.info("No saved clustering runs were found.")
        st.stop()

    st.caption(f"Results folder: {outputs_folder}")

    chosen_run = st.selectbox(
        "Saved clustering run",
        sorted(entries.keys()),
    )

    try:
        result = load_saved_run(
            outputs_folder,
            chosen_run,
        )

        table = result["table"]
        arrays = result["arrays"]
        settings = result["settings"]
        checkpoint = result["checkpoint_directory"]

        st.success(
            f"Loaded {len(table):,} particles from {chosen_run}."
        )

        # Saved plot
        st.subheader("Saved clustering plot")

        plot_path = checkpoint / "clusters.png"

        if plot_path.exists():
            st.image(
                str(plot_path),
                use_container_width=True,
            )
        else:
            st.info("No clusters.png was saved for this run.")

        # Cluster sizes
        if "cluster" in table.columns:

            st.subheader("Cluster sizes")

            cluster_counts = (
                table["cluster"]
                .value_counts()
                .sort_index()
                .rename("Particles")
            )

            st.dataframe(
                cluster_counts,
                use_container_width=True,
            )

        else:
            st.warning(
                "The saved table does not contain a 'cluster' column."
            )

        # Saved numerical scores
        st.subheader("Saved numerical results")

        if "selected_scores" in arrays:

            scores = arrays["selected_scores"]

            if scores.ndim == 2 and len(scores) == len(table):

                score_df = pd.DataFrame(
                    scores,
                    columns=[
                        f"Component {i + 1}"
                        for i in range(scores.shape[1])
                    ],
                )

                if "cluster" in table.columns:
                    score_df["cluster"] = (
                        table["cluster"].astype(str).to_numpy()
                    )

                    component_columns = [
                        column
                        for column in score_df.columns
                        if column != "cluster"
                    ]

                    if len(component_columns) >= 2:

                        left, right = st.columns(2)

                        with left:
                            x_axis = st.selectbox(
                                "Horizontal axis",
                                component_columns,
                                index=0,
                                key="score_x",
                            )

                        with right:
                            y_axis = st.selectbox(
                                "Vertical axis",
                                component_columns,
                                index=1,
                                key="score_y",
                            )

                        figure = px.scatter(
                            score_df,
                            x=x_axis,
                            y=y_axis,
                            color="cluster",
                            title=f"{chosen_run}: saved particle scores",
                        )

                        figure.update_layout(height=600)

                        st.plotly_chart(
                            figure,
                            use_container_width=True,
                        )

                    else:
                        st.info(
                            "At least two score components are needed "
                            "for a scatter plot."
                        )

                else:
                    st.info(
                        "The saved scores are available, but the table "
                        "has no cluster column for colouring."
                    )

            else:
                st.warning(
                    "The saved score array does not match the number "
                    "of rows in the clustering table."
                )

        else:
            st.info(
                "This run does not contain selected_scores. "
                "The saved clustering plot and table are still available."
            )

        # Full table
        st.subheader("Particle assignments")

        st.dataframe(
            table,
            use_container_width=True,
        )

        st.download_button(
            "Download saved assignments",
            table.to_csv(index=False).encode(),
            f"{chosen_run}_assignments.csv",
            "text/csv",
        )

        # Optional particle image restoration
        st.subheader("Particle images")

        st.caption(
            "Image restoration is optional. The numerical clustering "
            "results above do not require the original MRC files."
        )

        if st.button("Try to load particle images"):

            try:
                from cluster_examples import load_example_images

                input_directory = Path(
                    settings.get("input_directory", "")
                )

                if not input_directory.exists() and input_directory.name:
                    fallback = outputs_folder / input_directory.name

                    if fallback.exists():
                        input_directory = fallback

                input_bundle = None

                if settings.get("input_bundle_settings") is not None:
                    input_bundle = {
                        "settings": settings["input_bundle_settings"]
                    }

                images, image_dir, image_source = load_example_images(
                    outputs_folder,
                    table,
                    dataset="source",
                    fitted_source=settings.get("input_source"),
                    fitted_directory=input_directory,
                    fitted_bundle=input_bundle,
                )

                st.success("Particle images loaded.")
                st.caption(f"Image source: {image_source} · {image_dir}")

                if "cluster" not in table.columns:
                    st.warning(
                        "Cannot display cluster representatives because "
                        "the table has no cluster column."
                    )

                else:
                    labels = table["cluster"].to_numpy()
                    unique_labels = sorted(np.unique(labels))

                    st.subheader("Cluster representatives")

                    columns = st.columns(
                        min(4, max(1, len(unique_labels)))
                    )

                    for index, label in enumerate(unique_labels):

                        indices = np.flatnonzero(labels == label)

                        representative = int(
                            indices[len(indices) // 2]
                        )

                        with columns[index % len(columns)]:

                            st.markdown(
                                f"**Cluster {label}** · "
                                f"{len(indices)} particles"
                            )

                            image = image_from_array(
                                images[representative]
                            )

                            if image is not None:
                                st.image(
                                    image,
                                    caption=str(representative),
                                    use_container_width=True,
                                )

                    st.subheader("Inspect a particle")

                    particle_names = (
                        table["image_name"].astype(str).tolist()
                        if "image_name" in table.columns
                        else [str(i) for i in range(len(table))]
                    )

                    selected_particle = st.selectbox(
                        "Particle image ID",
                        particle_names,
                    )

                    row = particle_names.index(selected_particle)
                    label = labels[row]

                    cluster_indices = np.flatnonzero(labels == label)
                    representative = int(
                        cluster_indices[len(cluster_indices) // 2]
                    )

                    left, right = st.columns(2)

                    with left:
                        st.markdown(
                            f"**Selected particle · cluster {label}**"
                        )

                        image = image_from_array(images[row])

                        if image is not None:
                            st.image(
                                image,
                                use_container_width=True,
                            )

                    with right:
                        st.markdown(
                            f"**Representative of cluster {label}**"
                        )

                        image = image_from_array(
                            images[representative]
                        )

                        if image is not None:
                            st.image(
                                image,
                                use_container_width=True,
                            )

            except Exception as error:
                st.warning(
                    "The numerical results loaded successfully, but "
                    f"particle images could not be restored: {error}"
                )

    except Exception as error:
        st.error(f"Could not load saved clustering run: {error}")