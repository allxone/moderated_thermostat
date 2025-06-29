"""Config flow for Generic Thermostat."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import voluptuous as vol


from homeassistant.components.sensor import DOMAIN as SENSOR_DOMAIN, SensorDeviceClass
from homeassistant.const import CONF_NAME, PERCENTAGE
from homeassistant.helpers import selector
from homeassistant.helpers.schema_config_entry_flow import (
    SchemaConfigFlowHandler,
    SchemaFlowFormStep,
)
from homeassistant.components.generic_thermostat.config_flow import (
    CONFIG_SCHEMA as GENERIC_CONFIG_SCHEMA,
    OPTIONS_SCHEMA as GENERIC_OPTIONS_SCHEMA,
    PRESETS_SCHEMA as GENERIC_PRESETS_SCHEMA,
)
from .const import (
    CONF_LIMIT_HUM,
    CONF_SENSOR_HUM,
    DEFAULT_LIMIT_HUM,
    DOMAIN,
)

OPTIONS_SCHEMA = {
    **GENERIC_OPTIONS_SCHEMA,
    vol.Required(CONF_SENSOR_HUM): selector.EntitySelector(
        selector.EntitySelectorConfig(
            domain=SENSOR_DOMAIN, device_class=SensorDeviceClass.HUMIDITY
        )
    ),
    vol.Required(
        CONF_LIMIT_HUM, default=DEFAULT_LIMIT_HUM
    ): selector.NumberSelector(
        selector.NumberSelectorConfig(
            min=0,
            max=100,
            step=0.5,
            unit_of_measurement=PERCENTAGE,
            mode=selector.NumberSelectorMode.BOX,
        )
    ),
}

CONFIG_SCHEMA = {
    **GENERIC_CONFIG_SCHEMA,
    **OPTIONS_SCHEMA,
}

CONFIG_FLOW = {
    "user": SchemaFlowFormStep(vol.Schema(CONFIG_SCHEMA), next_step="presets"),
    "presets": SchemaFlowFormStep(vol.Schema(GENERIC_PRESETS_SCHEMA)),
}

OPTIONS_FLOW = {
    "init": SchemaFlowFormStep(vol.Schema(OPTIONS_SCHEMA), next_step="presets"),
    "presets": SchemaFlowFormStep(vol.Schema(GENERIC_PRESETS_SCHEMA)),
}

class ConfigFlowHandler(SchemaConfigFlowHandler, domain=DOMAIN):
    """Handle a config or options flow."""

    config_flow = CONFIG_FLOW
    options_flow = OPTIONS_FLOW

    def async_config_entry_title(self, options: Mapping[str, Any]) -> str:
        """Return config entry title."""
        return cast(str, options["name"])
