# 👟📍📈 Motion Map Analyzer

An interactive, web-based running data analytics platform built with Python, Streamlit, and Leaflet.js. This application parses standard `.tcx` and `.fit` fitness files, automatically smooths raw GPS/sensor noise, applies movement state classifications (running, walking, stopped), and renders a rich performance overlay directly onto a web map.

👉 **Try the live app here:** https://motion-map-app.streamlit.app/

---

## ✨ Features

* **Advanced Motion Segmentation:** Vectorized classification algorithms that divide your activity into highly precise Running, Walking, and Stopped segments.
* **Interactive Metric Overlays:** High-fidelity overlays for Pace, Heart Rate, and Cadence distributed seamlessly across map geometry.
* **Press-and-Hold Highlighting:** Smooth interactions enabling users to hold performance blocks (like Per-Km Splits or HR zones) to isolate exact route sections on the map.
* **Privacy Controls:** Optional on-the-fly privacy zones that trim the first and last 500 meters of your activity to mask sensitive start/end addresses.
* **Privacy-First Processing:** Secure architecture executing completely in RAM; all user data and analysis are processed in-memory and immediately destroyed post-session; zero retention.

---

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python 3.9+ installed on your system.

### 1. Clone the Repository
```bash
git clone [https://github.com/r777k/motion-map-app.git](https://github.com/r777k/motion-map-app.git)
cd motion-map-app