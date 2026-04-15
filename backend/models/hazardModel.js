const hazards = [];

function createHazard(payload) {
  const hazard = {
    id: hazards.length + 1,
    timestamp: new Date().toISOString(),
    ...payload,
  };
  hazards.push(hazard);
  return hazard;
}

function listHazards() {
  return hazards;
}

module.exports = {
  createHazard,
  listHazards,
};