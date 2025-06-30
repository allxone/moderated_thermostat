"""
Custom Moderated Thermostat climate platform for Home Assistant.

Extends the GenericThermostat to moderate temperature based on humidity limits.
"""

import logging
import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

import voluptuous as vol
from homeassistant.components.climate import (
    ATTR_HUMIDITY,
    PRESET_NONE,
    HVACMode,
)
from homeassistant.components.climate import (
    PLATFORM_SCHEMA as CLIMATE_PLATFORM_SCHEMA,
)
from homeassistant.components.generic_thermostat.climate import (
    CONF_INITIAL_HVAC_MODE,
    CONF_KEEP_ALIVE,
    CONF_PRECISION,
    CONF_TARGET_TEMP,
    CONF_TEMP_STEP,
    GenericThermostat,
)
from homeassistant.components.generic_thermostat.climate import (
    PLATFORM_SCHEMA_COMMON as GENERIC_PLATFORM_SCHEMA_COMMON,
)
from homeassistant.components.generic_thermostat.const import (
    CONF_AC_MODE,
    CONF_COLD_TOLERANCE,
    CONF_HEATER,
    CONF_HOT_TOLERANCE,
    CONF_MAX_TEMP,
    CONF_MIN_DUR,
    CONF_MIN_TEMP,
    CONF_PRESETS,
    CONF_SENSOR,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_NAME,
    CONF_UNIQUE_ID,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfTemperature,
)
from homeassistant.core import (
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import (
    AddConfigEntryEntitiesCallback,
    AddEntitiesCallback,
)
from homeassistant.helpers.event import (
    async_track_state_change_event,
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

from .const import CONF_LIMIT_HUM, CONF_SENSOR_HUM, DOMAIN, PLATFORMS

_LOGGER = logging.getLogger(__name__)
DEFAULT_NAME = "Moderated Thermostat"

PLATFORM_SCHEMA_COMMON = GENERIC_PLATFORM_SCHEMA_COMMON.extend(
    vol.Schema(
        {
            vol.Required(CONF_SENSOR_HUM): cv.entity_id,
            vol.Optional(CONF_LIMIT_HUM): vol.Coerce(float),
        }
    ).schema
)

PLATFORM_SCHEMA = CLIMATE_PLATFORM_SCHEMA.extend(PLATFORM_SCHEMA_COMMON.schema)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Initialize config entry."""
    await _async_setup_config(
        hass,
        PLATFORM_SCHEMA_COMMON(dict(config_entry.options)),
        config_entry.entry_id,
        async_add_entities,
    )


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,  # noqa: ARG001
) -> None:
    """Set up the moderated thermostat platform."""
    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)
    await _async_setup_config(
        hass, config, config.get(CONF_UNIQUE_ID), async_add_entities
    )


async def _async_setup_config(
    hass: HomeAssistant,
    config: Mapping[str, Any],
    unique_id: str | None,
    async_add_entities: AddEntitiesCallback | AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the moderated thermostat platform."""
    name: str = config[CONF_NAME]
    heater_entity_id: str = config[CONF_HEATER]
    sensor_entity_id: str = config[CONF_SENSOR]
    sensor_hum_entity_id: str = config[CONF_SENSOR_HUM]
    min_temp: float | None = config.get(CONF_MIN_TEMP)
    max_temp: float | None = config.get(CONF_MAX_TEMP)
    target_temp: float | None = config.get(CONF_TARGET_TEMP)
    limit_hum: float | None = config.get(CONF_LIMIT_HUM)
    ac_mode: bool | None = config.get(CONF_AC_MODE)
    min_cycle_duration: timedelta | None = config.get(CONF_MIN_DUR)
    cold_tolerance: float = config[CONF_COLD_TOLERANCE]
    hot_tolerance: float = config[CONF_HOT_TOLERANCE]
    keep_alive: timedelta | None = config.get(CONF_KEEP_ALIVE)
    initial_hvac_mode: HVACMode | None = config.get(CONF_INITIAL_HVAC_MODE)
    presets: dict[str, float] = {
        key: config[value] for key, value in CONF_PRESETS.items() if value in config
    }
    precision: float | None = config.get(CONF_PRECISION)
    target_temperature_step: float | None = config.get(CONF_TEMP_STEP)
    unit = hass.config.units.temperature_unit

    async_add_entities(
        [
            ModeratedThermostat(
                hass,
                name,
                heater_entity_id,
                sensor_entity_id,
                sensor_hum_entity_id,
                min_temp,
                max_temp,
                target_temp,
                limit_hum,
                ac_mode,
                min_cycle_duration,
                cold_tolerance,
                hot_tolerance,
                keep_alive,
                initial_hvac_mode,
                presets,
                precision,
                target_temperature_step,
                unit,
                unique_id,
            )
        ]
    )


class ModeratedThermostat(GenericThermostat):
    """
    Extended GenericThermostat to account for humidity limit.

    This thermostat controls the heating/cooling based on a humidity limit and the
    current humidity from a sensor.
    Any time current temperature and humidity change, the thermostat try to moderate
    target temperature to keep the humidity within the limit.
    It is useful if the cooler or heater is not able to control the humidity directly,
    but you want to keep it within a certain range sacrificing some temperature comfort.
    """

    _attr_should_poll = False

    def __init__(  # noqa: PLR0913
        self,
        hass: HomeAssistant,
        name: str,
        heater_entity_id: str,
        sensor_entity_id: str,
        sensor_hum_entity_id: str,
        min_temp: float | None,
        max_temp: float | None,
        target_temp: float | None,
        limit_hum: float | None,
        ac_mode: bool | None,  # noqa: FBT001
        min_cycle_duration: timedelta | None,
        cold_tolerance: float,
        hot_tolerance: float,
        keep_alive: timedelta | None,
        initial_hvac_mode: HVACMode | None,
        presets: dict[str, float],
        precision: float | None,
        target_temperature_step: float | None,
        unit: UnitOfTemperature,
        unique_id: str | None,
    ) -> None:
        """Initialize the thermostat."""
        self.sensor_hum_entity_id = sensor_hum_entity_id
        self._limit_hum: float | None = limit_hum
        self._cur_hum: float | None = None
        self._target_hum: float | None = None
        self._cur_moderation: float = 0

        # Save original cold tolerance and hot tolerance
        self._original_cold_tolerance = cold_tolerance
        self._original_hot_tolerance = hot_tolerance

        # Initialize the base class with the heater and sensor entities
        super().__init__(
            hass,
            name,
            heater_entity_id,
            sensor_entity_id,
            min_temp,
            max_temp,
            target_temp,
            ac_mode,
            min_cycle_duration,
            cold_tolerance,
            hot_tolerance,
            keep_alive,
            initial_hvac_mode,
            presets,
            precision,
            target_temperature_step,
            unit,
            unique_id,
        )

    @staticmethod
    def _calculate_saturation_vapor_pressure(temp_c: float) -> float:
        """
        Calculate the saturation vapor pressure (in kPa) using the Arden Buck equation.

        This formula is very accurate for temperatures between -40 and +50 °C.
        """
        numerator = (18.678 - (temp_c / 234.5)) * temp_c
        denominator = 257.14 + temp_c
        return 0.61121 * math.exp(numerator / denominator)

    @staticmethod
    def _predict_humidity(temp_c: float, hum_c: float, temp_t: float) -> float:
        """
        Predict humidity based on current temperature, humidity and target temperature.

        This method uses the saturation vapor pressure to calculate the predicted
        relative humidity at the target temperature.

        :param temp_c: Current temperature in Celsius.
        :param hum_c: Current relative humidity (0 to 1).
        :param temp_t: Target temperature in Celsius.
        :return: Predicted relative humidity at target temperature (0 to 1).
        """
        # Calculate saturation vapor pressure for current temperature
        saturation_pressure_current = (
            ModeratedThermostat._calculate_saturation_vapor_pressure(temp_c)
        )
        # Calculate actual vapor pressure based on current humidity
        actual_vapor_pressure = saturation_pressure_current * hum_c
        # Calculate saturation vapor pressure for target temperature
        saturation_pressure_target = (
            ModeratedThermostat._calculate_saturation_vapor_pressure(temp_t)
        )
        # Predict relative humidity at target temperature
        predicted_hum = round(actual_vapor_pressure / saturation_pressure_target, 3)

        # Ensure predicted humidity is not greater than 1
        return min(predicted_hum, 1.0)

    async def _async_moderate_temperature(self, step: float = 0.1) -> None:
        """Moderate target temperature based on current humidity and humidity limit."""
        # Reset moderation if conditions are not met
        if (
            self._cur_hum is None
            or self._limit_hum is None
            or self._cur_temp is None
            or self._target_temp is None
            or self._hvac_mode == HVACMode.OFF
            or not self._is_device_active
        ):
            self._cur_moderation = 0
            self._target_hum = None
            self._hot_tolerance = self._original_hot_tolerance
            self._cold_tolerance = self._original_cold_tolerance
            return

        # Decrease target temperature for heating mode
        if not self.ac_mode:
            step = step * -1

        # Derive source target temperature from target temp and current moderation
        self._cur_moderation = self._cur_moderation or 0
        moderation_c = step
        moderation_max = self._cur_temp - self._target_temp

        # Check if current humidity is within the limit
        if (self.ac_mode and self._cur_hum > self._limit_hum) or (
            not self.ac_mode and self._cur_hum < self._limit_hum
        ):
            self._cur_moderation = moderation_max
            self._target_hum = self._cur_hum
            self._hot_tolerance = self._original_hot_tolerance + (
                self._cur_moderation if not self.ac_mode else 0
            )
            self._cold_tolerance = self._original_cold_tolerance - (
                self._cur_moderation if self.ac_mode else 0
            )
            return

        # Predict humidity based on the current temperature, humidity and target temp
        predicted_hum = (
            ModeratedThermostat._predict_humidity(
                self._cur_temp, self._cur_hum / 100, self._target_temp
            )
            * 100
        )

        # If predicted_hum is within the limit, skip moderation
        if (self.ac_mode and predicted_hum < self._limit_hum) or (
            not self.ac_mode and predicted_hum > self._limit_hum
        ):
            self._cur_moderation = 0
            self._target_hum = predicted_hum
            self._hot_tolerance = self._original_hot_tolerance
            self._cold_tolerance = self._original_cold_tolerance
            return

        while abs(moderation_c) < abs(moderation_max):
            # Predict the humidity with the moderated target temperature
            predicted_hum = (
                self._predict_humidity(
                    self._cur_temp,
                    self._cur_hum / 100,
                    self._target_temp + moderation_c,
                )
                * 100
            )

            # If the predicted humidity is within the limit, set the moderation and
            # target humidity
            if (self.ac_mode and predicted_hum < self._limit_hum) or (
                not self.ac_mode and predicted_hum > self._limit_hum
            ):
                self._cur_moderation = moderation_c
                self._target_hum = predicted_hum
                self._hot_tolerance = self._original_hot_tolerance + (
                    self._cur_moderation if not self.ac_mode else 0
                )
                self._cold_tolerance = self._original_cold_tolerance - (
                    self._cur_moderation if self.ac_mode else 0
                )
                return

            # If not, increase the moderation step
            moderation_c += step

        # If we reach here, it means we couldn't moderate the temperature enough to
        # satisfy the humidity limit (moderation_max force the target temperature to
        # the current temperature)
        self._cur_moderation = moderation_max
        self._target_hum = predicted_hum
        self._hot_tolerance = self._original_hot_tolerance + (
            self._cur_moderation if not self.ac_mode else 0
        )
        self._cold_tolerance = self._original_cold_tolerance - (
            self._cur_moderation if self.ac_mode else 0
        )

    async def _async_control_heating(
        self,
        time: datetime | None = None,
        force: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Check if we need to turn heating on or off."""
        # Inject target temperature moderation into the control logic
        await self._async_moderate_temperature()

        # Call super to handle the heating control
        await super()._async_control_heating(time, force)

    @property
    def limit_humidity(self) -> float | None:
        """Return the humidity limit, if set."""
        return self._limit_hum

    @property
    def current_humidity(self) -> float | None:
        """Return current humidity."""
        return self._cur_hum

    @property
    def target_humidity(self) -> float | None:
        """Return predicted humidity."""
        return self._target_hum

    @property
    def current_moderation(self) -> float | None:
        """Return current humidity moderation applied to the target temperature."""
        return self._cur_moderation

    @property
    def moderated_temperature(self) -> float | None:
        """Return current humidity moderation applied to the target temperature."""
        if self._target_temp is not None:
            return self._cur_moderation + self._target_temp
        return None

    @property
    def moderation_active(self) -> bool:
        """Return current humidity moderation applied to the target temperature."""
        return self._cur_moderation != 0

    async def async_set_humidity(self, **kwargs: Any) -> None:
        """Set new humidity limit."""
        if (humidity := kwargs.get(ATTR_HUMIDITY)) is None:
            return
        self._attr_preset_mode = self._presets_inv.get(humidity, PRESET_NONE)
        self._target_hum = humidity
        await self._async_control_heating(force=True)
        self.async_write_ha_state()

    async def _async_sensor_hum_changed(
        self, event: Event[EventStateChangedData]
    ) -> None:
        """Handle humidity changes."""
        new_state = event.data["new_state"]
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        self._async_update_hum(new_state)
        await self._async_control_heating()
        self.async_write_ha_state()

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the state attributes."""
        attrs = dict(super().extra_state_attributes or {})
        attrs.update(
            {
                "limit_humidity": self.limit_humidity,
                "current_humidity": self.current_humidity,
                "target_humidity": self.target_humidity,
                "moderation_active": self.moderation_active,
                "moderated_temperature": self.moderated_temperature,
                "current_moderation": self.current_moderation,
            }
        )
        return attrs

    @callback
    def _async_update_hum(self, state: State) -> None:
        """Update current humidity with latest state from sensor_hum."""

        def _validate_humidity_state(state: State, cur_hum: float) -> None:
            if not math.isfinite(cur_hum) or cur_hum < 0 or cur_hum > 100:
                msg = f"Humidity Sensor has an illegal state {state.state}"
                raise ValueError(msg)

        try:
            cur_hum = float(state.state)
            _validate_humidity_state(state, cur_hum)
            self._cur_hum = cur_hum
        except ValueError:
            _LOGGER.exception("Unable to update from sensor")

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added."""
        await super().async_added_to_hass()

        # Add humidity listener
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.sensor_hum_entity_id], self._async_sensor_hum_changed
            )
        )
