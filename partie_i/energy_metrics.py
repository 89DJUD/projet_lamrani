"""Estimation CodeCarbon de la machine locale, incluant le serveur Ollama."""
import time
import math

class EnergyMonitor:
    def __init__(self, enabled=True, country_iso_code="MAR"):
        self.enabled = enabled
        self.country = country_iso_code
        self.tracker = None
        self.metrics = {"energy_kwh": None, "co2_kg": None,
                        "energy_method": "disabled", "energy_error": "",
                        "energy_scope": "local_machine_generation", "country_iso_code": self.country}

    def __enter__(self):
        self.start = time.perf_counter()
        if self.enabled:
            try:
                from codecarbon import OfflineEmissionsTracker
                self.tracker = OfflineEmissionsTracker(
                    country_iso_code=self.country, tracking_mode="machine",
                    measure_power_secs=1, save_to_file=False, save_to_api=False,
                    log_level="error")
                self.tracker.start()
                self.metrics["energy_method"] = "codecarbon_machine_estimate"
            except Exception as error:
                self.metrics.update(energy_method="unavailable", energy_error=str(error))
        # Exclut l'initialisation des capteurs du temps de generation.
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.metrics["seconds"] = time.perf_counter() - self.start
        if self.tracker is not None:
            try:
                emissions = self.tracker.stop()
                data = self.tracker.final_emissions_data
                energy = None if data is None else data.energy_consumed
                if emissions is None or energy is None:
                    raise RuntimeError("CodeCarbon n'a pas retourne de mesure.")
                if not all(math.isfinite(float(v)) and float(v) >= 0 for v in (energy, emissions)):
                    raise RuntimeError("Mesure energetique invalide.")
                self.metrics.update(energy_kwh=float(energy), co2_kg=float(emissions))
            except Exception as error:
                self.metrics.update(energy_method="unavailable", energy_error=str(error))
        return False
