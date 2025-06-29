"""Additional constants for the Moderated Thermostat helper."""

from homeassistant.const import Platform

DOMAIN = "moderated_thermostat"
PLATFORMS = [Platform.CLIMATE]

CONF_HEATER = "heater"
CONF_SENSOR_HUM = "target_sensor_hum"
CONF_LIMIT_HUM = "limit_hum"
DEFAULT_LIMIT_HUM = 60.0
