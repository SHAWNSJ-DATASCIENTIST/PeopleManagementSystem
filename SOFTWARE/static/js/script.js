const camImg = document.getElementById("camera-img");

function updateFrame() {
    camImg.src = "/latest_image?ts=" + Date.now();
}

async function updateStats() {
    try {
        const res = await fetch("/api/fusion_stats");
        if (!res.ok) return;

        const data = await res.json();

        const cam = data.camera || {};
        const piezo = data.piezo || {};
        const fusion = data.fusion || {};

        const camRisk = cam.risk || 0;
        const piezoRisk = piezo.risk || 0;
        const totalRisk = fusion.total_risk || 0;

        document.getElementById("camera-risk-bar").style.width = (camRisk * 100) + "%";
        document.getElementById("piezo-risk-bar").style.width = (piezoRisk * 100) + "%";
        document.getElementById("total-risk-bar").style.width = (totalRisk * 100) + "%";

        document.getElementById("camera-risk-val").innerText = camRisk.toFixed(2);
        document.getElementById("piezo-risk-val").innerText = piezoRisk.toFixed(2);
        document.getElementById("total-risk-val").innerText = totalRisk.toFixed(2);

        document.getElementById("conf-percent").innerText =
            Math.round((cam.confidence || 1) * 100) + "%";

        document.getElementById("vis-status").innerText =
            cam.status || "CLEAR";

        const raw = piezo.raw_data || {};

        document.getElementById("p1").innerText = raw.P1 || 0;
        document.getElementById("p2").innerText = raw.P2 || 0;
        document.getElementById("p3").innerText = raw.P3 || 0;
        document.getElementById("p4").innerText = raw.P4 || 0;
        document.getElementById("p5").innerText = raw.P5 || 0;

        document.getElementById("env-distance").innerText =
            raw.D !== undefined ? raw.D + " cm" : "-- cm";

        document.getElementById("env-vib").innerText = raw.V || "--";
        document.getElementById("env-ir").innerText = raw.IR || "--";
        document.getElementById("env-pir").innerText = raw.PIR || "--";

        document.getElementById("w-cam").innerText =
            fusion.weights?.cam ?? "0.6";

        document.getElementById("w-piezo").innerText =
            fusion.weights?.piezo ?? "0.4";

    } catch (e) {
        console.log("Stats error:", e);
    }
}

setInterval(updateFrame, 200);
setInterval(updateStats, 800);

updateFrame();
updateStats();