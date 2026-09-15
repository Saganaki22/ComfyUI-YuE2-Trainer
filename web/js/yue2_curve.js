// Frontend extension for the YuE2 Training Curve node.
//
// Two display paths:
//  1. FINAL — after the run, the curve node returns its polished chart via
//     `ui.images`; onExecuted turns that into node.imgs (same mechanism as
//     the built-in Preview Image node).
//  2. LIVE — while the trainer runs, it atomically rewrites a fixed PNG in
//     the temp folder every few seconds. As long as a prompt is executing we
//     poll that file and draw it into every YuE2 Training Curve node, so the
//     loss curve grows in real time. The final render replaces it at the end.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const LIVE_FILE = "yue2_curve_live.png";
const POLL_MS = 3000;
let pollTimer = null;

function viewUrl(filename, type) {
    const url = new URL("/view", window.location.origin);
    url.searchParams.set("filename", filename);
    url.searchParams.set("type", type ?? "temp");
    url.searchParams.set("subfolder", "");
    url.searchParams.set("t", Date.now().toString()); // cache-buster
    return url.toString();
}

function curveNodes() {
    return app.graph?._nodes?.filter((n) => n.type === "YuE2TrainingCurve") ?? [];
}

function showInCurveNodes(image) {
    for (const node of curveNodes()) {
        node.imgs = [image];
        if (node.size[0] < 420 || node.size[1] < 420) {
            node.setSize([Math.max(node.size[0], 420), Math.max(node.size[1], 420)]);
        }
        node.setDirtyCanvas(true, true);
    }
}

function pollLive() {
    if (!curveNodes().length) return;
    const image = new Image();
    image.onload = () => showInCurveNodes(image);
    image.onerror = () => {}; // trainer hasn't written the first frame yet
    image.src = viewUrl(LIVE_FILE);
}

function startPolling() {
    stopPolling();
    pollTimer = setInterval(pollLive, POLL_MS);
}

function stopPolling() {
    if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

app.registerExtension({
    name: "Starnodes.YuE2Trainer.TrainingCurve",

    setup() {
        api.addEventListener("execution_start", startPolling);
        api.addEventListener("execution_error", stopPolling);
        api.addEventListener("execution_interrupted", stopPolling);
        // The trainer node finishing = live feed is done; the curve node's own
        // onExecuted will immediately replace the live frame with the final chart.
        api.addEventListener("executed", (event) => {
            const node = app.graph?.getNodeById(event.detail?.node);
            if (node && (node.type === "YuE2LoRATrainer" || node.type === "YuE2ArtistARLoRATrainer" || node.type === "YuE2TrainingCurve")) {
                stopPolling();
            }
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "YuE2TrainingCurve") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const images = message?.images;
            if (!images || !images.length) return;

            stopPolling();
            const image = new Image();
            image.onload = () => showInCurveNodes(image);
            image.src = viewUrl(images[0].filename, images[0].type);
        };
    },
});
