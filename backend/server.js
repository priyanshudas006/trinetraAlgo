const express = require("express");
const hazardRoutes = require("./routes/hazardRoutes");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use("/api/hazards", hazardRoutes);

app.get("/", (req, res) => {
  res.status(200).json({
    status: "ok",
    service: "TRINETRA backend",
    message: "Use /health for status and /api/hazards for hazard ingestion",
  });
});

app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok", service: "TRINETRA backend" });
});

app.listen(PORT, () => {
  console.log(`TRINETRA backend running on port ${PORT}`);
});
