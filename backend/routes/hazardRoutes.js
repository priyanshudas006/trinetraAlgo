const express = require("express");
const { createHazard, listHazards } = require("../models/hazardModel");

const router = express.Router();

function isValidHazard(payload) {
  if (!payload || typeof payload !== "object") return false;
  const hasCoords =
    typeof payload.lat === "number" && Number.isFinite(payload.lat) &&
    typeof payload.lon === "number" && Number.isFinite(payload.lon);
  const validStatus = ["GREEN", "YELLOW", "RED"].includes(payload.status);
  return hasCoords && validStatus;
}

router.get("/", (req, res) => {
  res.status(200).json(listHazards());
});

router.post("/", (req, res) => {
  if (!isValidHazard(req.body)) {
    res.status(400).json({ error: "Invalid hazard payload" });
    return;
  }
  const hazard = createHazard(req.body || {});
  res.status(201).json(hazard);
});

module.exports = router;
