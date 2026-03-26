"""
Module for analyzing generated images,
extracting features, clustering, anomaly
detection, and classifier demonstration
using scikit-learn and visualization tools.
"""

import os
import argparse
import threading
import time
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis

try:
    from flask import Flask, request as flask_request, send_file, jsonify

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


SUPPORTED_IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif")
IMAGE_DIR = "data/images"
OUTPUT_CSV = "data/image_analysis.csv"


def run_classifier_example(df, feature_cols, label_col, out_dir):
    """
    Train/test a classifier using extracted features and
    cluster labels as pseudo-classes.
    """
    features = df[feature_cols].values
    labels = df[label_col].values
    (x_train, x_test, y_train, y_test) = train_test_split(
        features, labels, test_size=0.33, random_state=42, stratify=labels
    )
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classifier accuracy (RandomForest, label={label_col}): {acc:.2f}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix ({label_col})")
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{label_col}.png"))
    plt.close()


def analyze_image(image_path, save_hist_dir=None, features=None):
    """
        Analyze a single image and return statistics and features.
    def parse_args():
        parser = argparse.ArgumentParser(description="Analyze images and extract features.")
        parser.add_argument('--input-dir', type=str, default="data/images", help='Input image directory')
        parser.add_argument('--output-csv', type=str, default="data/image_analysis.csv", help='Output CSV file')
        parser.add_argument('--hist-dir', type=str, default=None, help='Directory to save histogram images (default: <input-dir>/histograms)')
        parser.add_argument('--no-hist', action='store_true', help='Disable saving histogram images')
        return parser.parse_args()
        sample of pixel values, unique values, and
        histogram features.
    """
    try:
        img = Image.open(image_path)
    except (IOError, OSError, UnidentifiedImageError) as e:
        print(f"Error opening {image_path}: {e}")
        return None

    # Color and grayscale analysis
    arr = np.array(img)
    stats = {"filename": os.path.basename(image_path)}
    # If image is grayscale, arr will be 2D; if color, 3D
    # Feature selection
    if features is not None:
        features = set(features)
    else:
        features = set()
    if arr.ndim == 2:
        # Grayscale
        arr_gray = arr
        stats.update(
            {
                k: v
                for k, v in {
                    "mean": float(np.mean(arr_gray)),
                    "std": float(np.std(arr_gray)),
                    "median": float(np.median(arr_gray)),
                    "skewness": float(skew(arr_gray.flatten())),
                    "kurtosis": float(kurtosis(arr_gray.flatten())),
                    "min": int(np.min(arr_gray)),
                    "max": int(np.max(arr_gray)),
                    "width": int(img.width),
                    "height": int(img.height),
                    "sample_pixels": arr_gray.flatten()[:10].tolist(),
                    "unique_values": int(len(np.unique(arr_gray))),
                }.items()
                if not features or k in features
            }
        )
        # Histogram (256 bins for grayscale)
        hist, _ = np.histogram(arr_gray, bins=256, range=(0, 255))
        if not features or "hist_sample" in features:
            stats["hist_sample"] = hist[:10].tolist()
        if not features or "hist_entropy" in features:
            stats["hist_entropy"] = float(
                -np.sum(
                    (hist / np.sum(hist) + 1e-8) * np.log2(hist / np.sum(hist) + 1e-8)
                )
            )
        # Save histogram plot if requested
        if save_hist_dir:
            plt.figure(figsize=(4, 2))
            plt.bar(range(256), hist, width=1, color="gray")
            plt.title(f"Histogram: {os.path.basename(image_path)}")
            plt.xlabel("Pixel Value")
            plt.ylabel("Count")
            plt.tight_layout()
            out_path = os.path.join(
                save_hist_dir, os.path.basename(image_path).replace(".png", "_hist.png")
            )
            plt.savefig(out_path)
            plt.close()
    else:
        # Color image (assume RGB)
        stats["width"] = int(img.width)
        stats["height"] = int(img.height)
        for i, channel in enumerate(["R", "G", "B"]):
            for k, v in {
                f"mean_{channel}": float(np.mean(arr[:, :, i])),
                f"std_{channel}": float(np.std(arr[:, :, i])),
                f"median_{channel}": float(np.median(arr[:, :, i])),
                f"min_{channel}": int(np.min(arr[:, :, i])),
                f"max_{channel}": int(np.max(arr[:, :, i])),
            }.items():
                if not features or k in features:
                    stats[k] = v
            # Histogram for each channel
            hist, _ = np.histogram(arr[:, :, i], bins=256, range=(0, 255))
            if not features or f"hist_sample_{channel}" in features:
                stats[f"hist_sample_{channel}"] = hist[:10].tolist()
            if not features or f"hist_entropy_{channel}" in features:
                stats[f"hist_entropy_{channel}"] = float(
                    -np.sum(
                        (hist / np.sum(hist) + 1e-8)
                        * np.log2(hist / np.sum(hist) + 1e-8)
                    )
                )
            if save_hist_dir:
                plt.figure(figsize=(4, 2))
                plt.bar(range(256), hist, width=1, color=channel.lower())
                plt.title(f"Histogram {channel}: {os.path.basename(image_path)}")
                plt.xlabel("Pixel Value")
                plt.ylabel("Count")
                plt.tight_layout()
                out_path = os.path.join(
                    save_hist_dir,
                    os.path.basename(image_path).replace(
                        ".png", f"_hist_{channel}.png"
                    ),
                )
                plt.savefig(out_path)
                plt.close()
        # Colorfulness metric (Hasler & Süsstrunk)
        rg = arr[:, :, 0] - arr[:, :, 1]
        yb = 0.5 * (arr[:, :, 0] + arr[:, :, 1]) - arr[:, :, 2]
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)
        if not features or "colorfulness" in features:
            stats["colorfulness"] = float(
                np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
            )
        if not features or "sample_pixels" in features:
            stats["sample_pixels"] = arr[:, :, 0].flatten()[:10].tolist()
        if not features or "unique_values" in features:
            stats["unique_values"] = int(len(np.unique(arr)))

    # Image quality metrics
    # Sharpness: variance of Laplacian
    from scipy.ndimage import laplace

    if not features or "sharpness" in features:
        if arr.ndim == 2:
            sharpness = float(np.var(laplace(arr)))
        else:
            arr_gray = np.array(img.convert("L"))
            sharpness = float(np.var(laplace(arr_gray)))
        stats["sharpness"] = sharpness
    if not features or "contrast" in features or "brightness" in features:
        if arr.ndim == 2:
            contrast = float(np.std(arr) / (np.mean(arr) + 1e-8))
            brightness = float(np.mean(arr))
        else:
            arr_gray = np.array(img.convert("L"))
            contrast = float(np.std(arr_gray) / (np.mean(arr_gray) + 1e-8))
            brightness = float(np.mean(arr_gray))
        if not features or "contrast" in features:
            stats["contrast"] = contrast
        if not features or "brightness" in features:
            stats["brightness"] = brightness
    return stats


def run_anomaly_detection(df, feature_cols, out_dir):
    """
    Run IsolationForest anomaly detection on selected features
    and save results.
    """
    features = df[feature_cols].values
    iso = IsolationForest(contamination=0.2, random_state=42)
    preds = iso.fit_predict(features)
    df["anomaly"] = preds
    # -1 means anomaly, 1 means normal
    anomalies = df[df["anomaly"] == -1]
    print("Anomalies detected:")
    print(anomalies[["filename"] + feature_cols + ["cluster", "pca1", "pca2"]])
    df.to_csv(os.path.join(out_dir, "image_analysis_with_anomaly.csv"), index=False)
    # Visualize anomalies in PCA space
    plt.figure(figsize=(6, 4))
    plt.scatter(
        df["pca1"],
        df["pca2"],
        c=(df["anomaly"] == -1),
        cmap="coolwarm",
        s=60,
        label="Normal/Anomaly",
    )
    for _, row in df.iterrows():
        if row["anomaly"] == -1:
            plt.text(row["pca1"], row["pca2"], row["filename"], fontsize=8)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Anomaly Detection (IsolationForest)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "anomaly_pca.png"))
    plt.close()
    print(
        "Anomaly detection results saved to "
        f"{os.path.join(out_dir, 'image_analysis_with_anomaly.csv')}"
    )
    print("Anomaly PCA plot saved to " f"{os.path.join(out_dir, 'anomaly_pca.png')}")


def collect_image_stats(image_dir, supported_formats, hist_dir, features=None):
    image_files = [
        f for f in os.listdir(image_dir) if f.lower().endswith(supported_formats)
    ]
    results = []
    for fname in tqdm(sorted(image_files), desc="Analyzing images", unit="img"):
        path = os.path.join(image_dir, fname)
        stats = analyze_image(path, save_hist_dir=hist_dir, features=features)
        if stats is not None:
            results.append(stats)
    return results


def perform_clustering_and_pca(df, feature_cols, image_dir):
    features = df[feature_cols].values
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features)
    df["cluster"] = clusters
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    df["pca1"] = features_pca[:, 0]
    df["pca2"] = features_pca[:, 1]
    df.to_csv(OUTPUT_CSV, index=False)
    # Scatter plot: mean vs std, colored by cluster
    plt.figure(figsize=(6, 4))
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        plt.scatter(sub["mean"], sub["std"], label=f"Cluster {c}", s=60)
    plt.xlabel("Mean Pixel Value")
    plt.ylabel("Std Dev Pixel Value")
    plt.title("Image Clusters (Mean vs Std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "cluster_scatter.png"))
    plt.close()
    # PCA scatter plot
    plt.figure(figsize=(6, 4))
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        plt.scatter(sub["pca1"], sub["pca2"], label=f"Cluster {c}", s=60)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Image Clusters (PCA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "cluster_pca.png"))
    plt.close()
    print("Cluster and PCA visualizations saved in data/images/")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze images and extract features.")
    parser.add_argument(
        "--input-dir", type=str, default="data/images", help="Input image directory"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/image_analysis.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--hist-dir",
        type=str,
        default=None,
        help="Directory to save histogram images (default: <input-dir>/histograms)",
    )
    parser.add_argument(
        "--no-hist", action="store_true", help="Disable saving histogram images"
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated list of features to compute (default: all)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="csv",
        choices=["csv", "json", "excel"],
        help="Output format for results",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=3, help="Number of clusters for KMeans"
    )
    parser.add_argument(
        "--anomaly-contamination",
        type=float,
        default=0.2,
        help="Contamination for anomaly detection",
    )
    parser.add_argument(
        "--email-report",
        type=str,
        default=None,
        help="Email address to send report to (optional)",
    )
    parser.add_argument(
        "--slack-webhook",
        type=str,
        default=None,
        help="Slack webhook URL to notify (optional)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Monitor directory for new images and auto-analyze",
    )
    parser.add_argument(
        "--api", action="store_true", help="Run as REST API server (Flask required)"
    )
    return parser.parse_args()


def send_email_report(report_path, to_email):
    # Simple SMTP email sender (customize as needed)
    from_addr = "noreply@imageanalyzer.local"
    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["To"] = to_email
    msg["Subject"] = "Image Analysis Report"
    with open(report_path, "r", encoding="utf-8") as f:
        html = f.read()
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP("localhost") as server:
            server.sendmail(from_addr, to_email, msg.as_string())
        print(f"Report emailed to {to_email}")
    except Exception as e:
        print(f"[Warning] Could not send email: {e}")


def notify_slack(webhook_url, message):
    try:
        resp = requests.post(webhook_url, json={"text": message})
        if resp.status_code == 200:
            print("Slack notification sent.")
        else:
            print(f"Slack notification failed: {resp.text}")
    except Exception as e:
        print(f"[Warning] Could not notify Slack: {e}")


def generate_html_report(df, input_dir, output_csv):
    import base64
    from io import BytesIO

    html_path = os.path.splitext(output_csv)[0] + "_report.html"
    html = [
        "<html><head><title>Image Analysis Report</title><style>body{font-family:sans-serif;} table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:4px;} img{max-width:120px;max-height:120px;}</style></head><body>"
    ]
    html.append(
        f"<h1>Image Analysis Report</h1><p>Input directory: <b>{input_dir}</b></p><p>Output CSV: <b>{output_csv}</b></p>"
    )
    html.append(
        "<table><tr><th>Image</th><th>Filename</th><th>Mean</th><th>Std</th><th>Contrast</th><th>Sharpness</th><th>Brightness</th></tr>"
    )
    for _, row in df.iterrows():
        img_path = os.path.join(input_dir, row["filename"])
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as im:
                    buf = BytesIO()
                    im.thumbnail((120, 120))
                    im.save(buf, format="PNG")
                    img_b64 = base64.b64encode(buf.getvalue()).decode()
                    img_tag = f'<img src="data:image/png;base64,{img_b64}" />'
            except Exception:
                img_tag = "(error)"
        else:
            img_tag = "(missing)"
        html.append(
            f"<tr><td>{img_tag}</td><td>{row['filename']}</td><td>{row.get('mean','')}</td><td>{row.get('std','')}</td><td>{row.get('contrast','')}</td><td>{row.get('sharpness','')}</td><td>{row.get('brightness','')}</td></tr>"
        )
    html.append("</table>")
    # Add cluster and PCA plots if available
    for plot_name in ["cluster_scatter.png", "cluster_pca.png", "anomaly_pca.png"]:
        plot_path = os.path.join(input_dir, plot_name)
        if os.path.exists(plot_path):
            with open(plot_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
                html.append(
                    f'<h3>{plot_name}</h3><img src="data:image/png;base64,{img_b64}" />'
                )
    html.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"HTML report saved to {html_path}")
    return html_path


def analyze_and_report(args):
    input_dir = args.input_dir
    output_csv = args.output_csv
    hist_dir = args.hist_dir if args.hist_dir else os.path.join(input_dir, "histograms")
    save_hist = not args.no_hist
    features = args.features.split(",") if args.features else None
    output_format = args.output_format
    n_clusters = args.n_clusters
    anomaly_contamination = args.anomaly_contamination

    if save_hist:
        os.makedirs(hist_dir, exist_ok=True)
    results = collect_image_stats(
        input_dir,
        SUPPORTED_IMAGE_FORMATS,
        hist_dir if save_hist else None,
        features=features,
    )
    df = pd.DataFrame(results)
    print(df)
    if output_format == "csv":
        df.to_csv(output_csv, index=False)
    elif output_format == "json":
        df.to_json(output_csv.replace(".csv", ".json"), orient="records", indent=2)
    elif output_format == "excel":
        df.to_excel(output_csv.replace(".csv", ".xlsx"), index=False)
    print(f"Analysis saved to {output_csv}")
    if save_hist:
        print(f"Histogram visualizations saved in {hist_dir}")

    # --- Advanced Feature Engineering and Visualization ---
    feature_cols = [
        c for c in ["mean", "std", "unique_values", "hist_entropy"] if c in df.columns
    ]
    if feature_cols:
        df = perform_clustering_and_pca(df, feature_cols, input_dir)
        run_anomaly_detection(df, feature_cols, input_dir)
        if "cluster" in df.columns:
            run_classifier_example(df, feature_cols, "cluster", input_dir)
    else:
        print(
            "Advanced features not available in CSV. Please rerun analysis with updated script."
        )

    html_path = generate_html_report(df, input_dir, output_csv)
    if args.email_report:
        send_email_report(html_path, args.email_report)
    if args.slack_webhook:
        notify_slack(
            args.slack_webhook, f"Image analysis complete. Report: {html_path}"
        )


def watch_directory(args):
    print(f"Watching directory {args.input_dir} for new images...")
    seen = set(os.listdir(args.input_dir))
    while True:
        time.sleep(5)
        current = set(os.listdir(args.input_dir))
        new_files = [
            f for f in current - seen if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)
        ]
        if new_files:
            print(f"New images detected: {new_files}")
            analyze_and_report(args)
        seen = current


def run_api_server(args):
    if not FLASK_AVAILABLE:
        print("Flask is not installed. Please install Flask to use the API server.")
        return
    app = Flask(__name__)

    @app.route("/analyze", methods=["POST"])
    def analyze_endpoint():
        data = flask_request.json or {}

        class Args:
            pass

        for k, v in data.items():
            setattr(Args, k, v)
        # Fill missing args with defaults
        for k, v in vars(parse_args()).items():
            if not hasattr(Args, k):
                setattr(Args, k, v)
        analyze_and_report(Args)
        return jsonify({"status": "ok"})

    @app.route("/report", methods=["GET"])
    def get_report():
        args = parse_args()
        html_path = os.path.splitext(args.output_csv)[0] + "_report.html"
        if os.path.exists(html_path):
            return send_file(html_path)
        return "Report not found", 404

    app.run(host="0.0.0.0", port=5000)


def main():
    args = parse_args()
    if args.api:
        run_api_server(args)
        return
    if args.watch:
        t = threading.Thread(target=watch_directory, args=(args,), daemon=True)
        t.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopped watching.")
    else:
        analyze_and_report(args)


if __name__ == "__main__":
    main()
