from pathlib import Path
import io, zipfile, importlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

st.set_page_config(page_title="Cryo-EM particle explorer", page_icon="🔬", layout="wide")
st.title("🔬 Cryo-EM particle latent-space explorer")
st.caption("Explore particle representations, clustering and representative images. Cluster IDs are unsupervised labels—not biological dimer/tetramer diagnoses.")

def image_from_array(a):
    a = np.asarray(a)
    if a.ndim == 1:
        side = int(np.sqrt(a.size))
        if side * side == a.size: a = a.reshape(side, side)
        else: return None
    if a.ndim == 3 and a.shape[0] in (1, 3, 4): a = np.moveaxis(a, 0, -1)
    if a.ndim == 3 and a.shape[-1] not in (1, 3, 4): a = a[0]
    a = np.squeeze(a)
    if a.ndim != 2 and not (a.ndim == 3 and a.shape[-1] in (1,3,4)): return None
    a = a.astype(float)
    lo, hi = np.nanpercentile(a, [1, 99])
    a = np.clip((a-lo) / max(hi-lo, 1e-12), 0, 1)
    return Image.fromarray((a*255).astype(np.uint8))

def read_image_zip(upload):
    if upload is None: return {}
    out = {}
    with zipfile.ZipFile(upload) as z:
        for n in z.namelist():
            if n.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff",".bmp")):
                try:
                    im = Image.open(io.BytesIO(z.read(n))).convert("L")
                    out[Path(n).name] = im.copy()
                    out[Path(n).stem] = im.copy()
                except Exception: pass
    return out

def find_image(images, name):
    if not images: return None
    key = str(name)
    return images.get(key) or images.get(Path(key).name) or images.get(Path(key).stem)

def numeric_matrix(df, excluded):
    cols = [c for c in df.select_dtypes(include=np.number).columns if c not in excluded]
    if not cols: raise ValueError("No numeric feature columns found. Keep an ID column and numeric feature columns in the CSV.")
    x = df[cols].replace([np.inf,-np.inf], np.nan)
    x = x.fillna(x.median()).fillna(0).to_numpy(dtype=float)
    if x.shape[0] < 3: raise ValueError("At least three particles are needed.")
    return x, cols

def reduce_space(x, method, seed, neighbors, min_dist):
    z = StandardScaler().fit_transform(x)
    if method == "PCA":
        n = min(3, z.shape[0]-1, z.shape[1])
        model = PCA(n_components=n, random_state=seed)
        emb = model.fit_transform(z)
        variance = float(model.explained_variance_ratio_[:2].sum())
        return emb, f"PCA (PC1+PC2 explain {variance:.1%} of variance)"
    if method == "ICA":
        n = min(3, z.shape[0]-1, z.shape[1])
        emb = FastICA(n_components=n, random_state=seed, max_iter=1000, tol=0.001).fit_transform(z)
        return emb, "ICA independent components"
    try:
        import umap
    except ImportError as e:
        raise RuntimeError("UMAP is missing. Install umap-learn from requirements.") from e
    n_neighbors = min(neighbors, len(z)-1)
    emb = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed).fit_transform(z)
    return emb, "UMAP"
    
def cluster_data(x, method, k, eps, min_samples, min_cluster):
    if method == "K-means":
        return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(x)
    if method == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
    try:
        import hdbscan
    except ImportError as e:
        raise RuntimeError("HDBSCAN is missing. Install hdbscan from requirements.") from e
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samples).fit_predict(x)

def representative_index(x, idx):
    subset = x[idx]
    center = subset.mean(axis=0, keepdims=True)
    return int(idx[np.argmin(pairwise_distances(subset, center).ravel())])

def plot_embedding(emb, labels, ids, method_label):
    data = pd.DataFrame({"x":emb[:,0], "y":emb[:,1], "cluster":labels.astype(str), "particle":ids})
    if emb.shape[1] > 2: data["z"] = emb[:,2]
    if emb.shape[1] > 2:
        fig = px.scatter_3d(data, x="x", y="y", z="z", color="cluster", hover_name="particle", title=method_label)
    else:
        fig = px.scatter(data, x="x", y="y", color="cluster", hover_name="particle", title=method_label)
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

with st.sidebar:
    st.header("Data")
    mode = st.radio("Input source", ["Feature CSV (flexible)", "Saved notebook clustering results"])
    img_zip = st.file_uploader("Optional particle image archive (.zip)", type=["zip"])
    image_map = read_image_zip(img_zip)
    st.caption("Image archive filenames should match particle IDs in the table.")

if mode == "Feature CSV (flexible)":
    csv_file = st.file_uploader("Upload particle feature table (.csv)", type=["csv"])
    if csv_file is None:
        st.info("Upload a CSV with one row per particle, an ID column (particle_id or image_name), and numeric feature columns. Optionally include a precomputed cluster column.")
        st.stop()
    df = pd.read_csv(csv_file)
    id_candidates = [c for c in ("particle_id","image_name","image_id","id") if c in df.columns]
    id_col = st.selectbox("Particle ID column", id_candidates or list(df.columns))
    ids = df[id_col].astype(str).to_numpy()
    existing = next((c for c in ("cluster","cluster_id","labels") if c in df.columns), None)
    excluded = {id_col}
    if existing: excluded.add(existing)
    if "known_class" in df.columns: excluded.add("known_class")
    x, feature_cols = numeric_matrix(df, excluded)
    with st.sidebar:
        st.header("Analysis")
        reducer = st.selectbox("Embedding", ["PCA","ICA","UMAP"])
        cluster_method = st.selectbox("Clustering", ["K-means","DBSCAN","HDBSCAN"])
        cluster_space = st.selectbox("Cluster on", ["Scaled features","2D/3D embedding"])
        k = st.slider("K-means clusters", 2, min(20, max(2,len(df)-1)), min(4,max(2,len(df)-1)))
        eps = st.slider("DBSCAN eps", 0.05, 5.0, 0.7, 0.05)
        min_samples = st.slider("Minimum samples", 2, 30, 5)
        min_cluster = st.slider("HDBSCAN minimum cluster size", 2, min(50,max(2,len(df))), min(10,max(2,len(df))))
        seed = st.number_input("Random seed", 0, 100000, 42)
        neighbors = st.slider("UMAP neighbors", 2, min(50,max(2,len(df)-1)), min(15,max(2,len(df)-1)))
        min_dist = st.slider("UMAP min_dist", 0.0, 0.99, 0.1)
    try:
        emb, emb_label = reduce_space(x, reducer, int(seed), neighbors, min_dist)
        cluster_input = StandardScaler().fit_transform(x) if cluster_space == "Scaled features" else emb
        labels = df[existing].fillna(-1).astype(int).to_numpy() if existing else cluster_data(cluster_input, cluster_method, k, eps, min_samples, min_cluster)
    except Exception as e:
        st.error(str(e)); st.stop()
    df_out = df.copy()
    df_out["app_cluster"] = labels
    st.subheader("Latent space")
    plot_embedding(emb, labels, ids, emb_label + " · " + (existing or cluster_method))
    counts = pd.Series(labels).value_counts().sort_index()
    a,b,c = st.columns(3)
    a.metric("Particles", len(df))
    b.metric("Clusters (excluding noise)", int((counts.index != -1).sum()))
    c.metric("Noise points", int(counts.get(-1,0)))
    st.subheader("Cluster representatives")
    reps = {}
    cols = st.columns( min(4, max(1, len(counts))) )
    for j, lab in enumerate(sorted(np.unique(labels))):
        idx = np.flatnonzero(labels == lab)
        rep = representative_index(x, idx)
        reps[lab] = rep
        with cols[j % len(cols)]:
            st.markdown(f"**Cluster {lab}** · {len(idx)} particles")
            im = find_image(image_map, ids[rep])
            if im: st.image(im, caption=f"Representative: {ids[rep]}", use_container_width=True)
            else: st.caption(f"Representative particle: {ids[rep]} (no matching image in ZIP)")
    st.subheader("Inspect a cluster")
    chosen = st.selectbox("Cluster", sorted(np.unique(labels)), format_func=lambda v: f"Cluster {v} ({int((labels==v).sum())} particles)")
    members = np.flatnonzero(labels == chosen)
    st.dataframe(df_out.iloc[members].head(500), use_container_width=True)
    st.download_button("Download assignments CSV", df_out.to_csv(index=False).encode(), "particle_cluster_assignments.csv", "text/csv")
    st.subheader("Find a particle")
    particle = st.selectbox("Existing particle", ids.tolist())
    row = int(np.flatnonzero(ids == particle)[0])
    lab = labels[row]
    left,right = st.columns(2)
    with left:
        st.markdown(f"**Particle {particle} · cluster {lab}**")
        im = find_image(image_map, particle)
        if im: st.image(im, use_container_width=True)
        else: st.info("No image matched this particle ID in the uploaded ZIP.")
    with right:
        rep = reps[lab]
        st.markdown(f"**Representative of cluster {lab}**")
        rim = find_image(image_map, ids[rep])
        if rim: st.image(rim, use_container_width=True)
        else: st.info(f"Representative ID: {ids[rep]}")
    st.markdown("**Upload a new particle**")
    new_img = st.file_uploader("New particle image", type=["png","jpg","jpeg","tif","tiff"], key="new_particle")
    if new_img:
        try:
            pipeline = importlib.import_module("app_feature_pipeline")
            new_features = np.asarray(pipeline.extract_features(Image.open(new_img).convert("L")), dtype=float).reshape(1,-1)
            if new_features.shape[1] != x.shape[1]:
                st.error(f"Feature pipeline returned {new_features.shape[1]} values; expected {x.shape[1]}.")
            else:
                scaler = StandardScaler().fit(x)
                dist = pairwise_distances(scaler.transform(new_features), scaler.transform(x)).ravel()
                nearest = int(np.argmin(dist))
                assigned = labels[nearest]
                st.success(f"Nearest-reference assignment: cluster {assigned}")
                st.caption(f"Nearest particle: {ids[nearest]} · standardized feature distance: {dist[nearest]:.4f}. This is a nearest-neighbour lookup, not a calibrated probability.")
                l,r = st.columns(2)
                with l: st.image(Image.open(new_img), caption="Uploaded particle", use_container_width=True)
                with r:
                    st.markdown(f"**Representative of cluster {assigned}**")
                    ri = find_image(image_map, ids[reps[assigned]])
                    if ri: st.image(ri, use_container_width=True)
                    else: st.write(f"Representative ID: {ids[reps[assigned]]}")
        except ModuleNotFoundError:
            st.warning("New-image assignment is not enabled yet. Add app_feature_pipeline.py with extract_features(image), using exactly the same preprocessing and feature order as the CSV. Existing-particle lookup works now.")
        except Exception as e: st.error(f"Could not process uploaded image: {e}")

else:
    st.info("This mode reads the saved clustering checkpoints created by the notebook in LatentSpace_data. Run the notebook's save-clustering section first, then launch this app from the project root.")
    project_root = None
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / "LatentSpace_data").is_dir():
            project_root = p
            break
    if project_root is None:
        st.warning("Could not find a LatentSpace_data folder above the current working directory.")
        st.stop()
    data_root = project_root / "LatentSpace_data"
    index_file = data_root / "clustering_index.json"
    if not index_file.exists():
        st.warning("No clustering_index.json found. Save at least one named clustering result in the notebook first.")
        st.stop()
    import json
    entries = json.loads(index_file.read_text())
    if not entries:
        st.warning("No saved clustering runs found."); st.stop()
    names = sorted(entries)
    chosen_run = st.selectbox("Saved clustering run", names)
    try:
        from analysis_storage import load_clustering
        from cluster_examples import load_example_images
        result = load_clustering(data_root, chosen_run)
        table = result["table"].copy()
        st.success(f"Loaded {len(table)} rows from saved run: {chosen_run}")
        labels = table["cluster"].to_numpy()
        st.dataframe(table, use_container_width=True)
        # Restore the exact source images associated with this clustering table.
        try:
            images, image_dir, image_source = load_example_images(
                data_root, table, dataset="source",
                fitted_source=result["input_source"],
                fitted_directory=result["input_directory"],
                fitted_bundle=result.get("input_bundle"))
            st.caption(f"Images: {image_source} · {image_dir}")
            st.subheader("Cluster representatives")
            for lab in sorted(np.unique(labels)):
                idx = np.flatnonzero(labels == lab)
                rep_local = int(idx[len(idx)//2])
                cols = st.columns( min(4, max(1, len(np.unique(labels)))) )
                with cols[list(sorted(np.unique(labels))).index(lab) % len(cols)]:
                    st.markdown(f"**Cluster {lab}** · {len(idx)} particles")
                    pic = image_from_array(images[rep_local])
                    if pic: st.image(pic, caption=str(table.iloc[rep_local]["image_name"]), use_container_width=True)
            st.subheader("Inspect particle")
            particle_names = table["image_name"].astype(str).tolist()
            selected = st.selectbox("Particle image ID", particle_names)
            row = particle_names.index(selected)
            lab = labels[row]
            rep_idx = int(np.flatnonzero(labels == lab)[len(np.flatnonzero(labels == lab))//2])
            left,right = st.columns(2)
            with left:
                st.markdown(f"**Selected particle · cluster {lab}**")
                pic = image_from_array(images[row])
                if pic: st.image(pic, use_container_width=True)
            with right:
                st.markdown(f"**Representative of cluster {lab}**")
                pic = image_from_array(images[rep_idx])
                if pic: st.image(pic, use_container_width=True)
            st.caption("Saved-run mode performs exact lookup for existing particles. Uploading a new image requires an app_feature_pipeline.py implementation matching the notebook preprocessing and fitted feature transform.")
        except Exception as e:
            st.warning(f"Could not restore particle images for this saved run: {e}. You can still inspect the cluster table.")
        st.download_button("Download saved assignments", table.to_csv(index=False).encode(), f"{chosen_run}_assignments.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not load saved run: {e}")
