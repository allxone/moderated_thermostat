"""Additional constants for the Moderated Thermostat helper."""

from homeassistant.const import Platform

DOMAIN = "moderated_thermostat"

PLATFORMS = [Platform.CLIMATE]

CONF_SENSOR_HUM = "target_sensor_hum"