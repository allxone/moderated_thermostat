import logging

from homeassistant.components.generic_thermostat.climate import GenericThermostat
from homeassistant.components.generic_thermostat.climate import async_setup_platform as generic_async_setup_platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
import math

from .const import (
    CONF_SENSOR_HUM,
    DOMAIN,
    PLATFORMS,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = "Moderated Thermostat"

CONF_LIMIT_HUM = "limit_hum"



# Riusiamo la funzione di setup originale per gestire la configurazione YAML,
# ma le passiamo un wrapper per intercettare le entità e sostituirle
# con la nostra classe estesa.
async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    """Set up the Moderated thermostat platform."""
    _LOGGER.info("Setting up Moderated Thermostat platform")

    def add_entities_wrapper(entities, update_before_add=False):
        """Wrapper per sostituire la classe originale con la nostra."""
        extended_entities = []
        for entity in entities:
            # Ci assicuriamo che l'entità sia quella che vogliamo estendere
            if isinstance(entity, GenericThermostat):
                _LOGGER.debug("Wrapping GenericThermostat entity: %s", entity.name)
                # Creiamo un'istanza della nostra classe estesa.
                # Per farlo, dobbiamo purtroppo accedere ad alcuni attributi
                # protetti (_config) dell'istanza originale.
                extended_entities.append(ModeratedThermostat(hass, entity._config))
            else:
                extended_entities.append(entity) # Aggiungi altre eventuali entità non modificate

        # Passa la lista delle entità (ora con la nostra classe) alla funzione originale
        async_add_entities(extended_entities, update_before_add)


    # Chiama la funzione di setup della piattaforma originale, ma fornendo
    # il nostro wrapper al posto della vera funzione async_add_entities.
    await generic_async_setup_platform(hass, config, add_entities_wrapper, discovery_info)


class ModeratedThermostat(GenericThermostat):
    """Extended GenericThermostat to account for humidity limit.
    This thermostat controls the heating/cooling based on a humidity limit and the current humidity from a sensor.
    Any time current temperature and humidity change, the thermostat try to moderate target temperature to keep the humidity within the limit.
    It is useful if the cooler or heater is not able to control the humidity directly, but you want to keep it within a certain range sacrificing some temperature comfort.
    """

    def _calculate_saturation_vapor_pressure(self, temp_c: float) -> float:
        """
        Calcola la pressione di vapore saturo (in kPa) usando l'equazione di Arden Buck.
        Questa formula è molto accurata per temperature tra -40 e +50 °C.
        """
        numerator = (18.678 - (temp_c / 234.5)) * temp_c
        denominator = 257.14 + temp_c
        return 0.61121 * math.exp(numerator / denominator)
    
    def _predict_humidity(self, temp_c: float, hum_c: float, temp_t: float) -> float:
        """Predict relative humidity based on current temperature, humidity and target temperature by Arden Buck equation."""
        if temp_c is None or hum_c is None or temp_t is None:
            return None

        # Calculate saturation vapor pressure for current temperature
        saturation_pressure_current = self._calculate_saturation_vapor_pressure(temp_c)
        # Calculate actual vapor pressure based on current humidity
        actual_vapor_pressure = saturation_pressure_current * self.hum_c
        # Calculate saturation vapor pressure for target temperature
        saturation_pressure_target = self._calculate_saturation_vapor_pressure(temp_t)
        # Predict relative humidity at target temperature
        predicted_hum = actual_vapor_pressure / saturation_pressure_target

        return min(predicted_hum, 1)  # Ensure predicted humidity is not greater than 1


    async def _async_moderate_temperature(self, step: float = 0.1) -> None:
        """ Moderates the target temperature based on the current humidity and the humidity limit."""
        if self._cur_hum is None or self._limit_hum is None or self._hvac_mode == HVACMode.OFF or not self._is_device_active:
            self._cur_moderation = 0
            self._target_hum = None
            return

        # Decrease target temperature for heating mode
        if not self.ac_mode:
            step = step * -1

        # Derive original target temperature from the current target temperature and the current moderation
        self._cur_moderation = self._cur_moderation or 0
        target_temp = self._target_temp - self._cur_moderation

        # Predict humidity based on the current temperature, humidity and target temperature
        predicted_hum = self._predict_humidity(self._cur_temp, self._cur_hum, target_temp)

        # If predicted_hum is within the limit, skip moderation
        if  (self.ac_mode and predicted_hum < self._limit_hum) or (not self.ac_mode and predicted_hum > self._limit_hum):
            self._target_temp = target_temp
            self._cur_moderation = 0
            self._target_hum = predicted_hum
            return

        # predicted_hum outside the limit, moderate the target temperature
        moderation_c = step
        moderation_max = target_temp - self._cur_temp

        while abs(moderation_c) < abs(moderation_max):

            # Predict the humidity with the moderated target temperature
            predicted_hum = self._predict_humidity(self._cur_temp, self._cur_hum, self._target_temp + moderation_c)

            # If the predicted humidity is within the limit, set the moderation and target humidity
            if  (self.ac_mode and predicted_hum < self._limit_hum) or (not self.ac_mode and predicted_hum > self._limit_hum):
                self._target_temp = target_temp + moderation_c
                self._cur_moderation = moderation_c
                self._target_hum = predicted_hum
                return

            # If not, increase the moderation step
            moderation_c += step            

        # If we reach here, it means we couldn't moderate the temperature enough to satisfy the humidity limit
        # (moderation_max force the target temperature to the current temperature)
        self._target_temp = target_temp + moderation_max
        self._cur_moderation = moderation_max
        self._target_hum = predicted_hum


    async def _async_control_heating(
        self, time: datetime | None = None, force: bool = False
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
        """Return the sensor_hum humidity."""
        return self._cur_hum

    @property
    def target_humidity(self) -> float | None:
        """Return predicted humidity if target temperature is reached."""
        return self._target_hum

    @property
    def current_moderation(self) -> float | None:
        """Return current humidity moderation applied to the target temperature."""
        return self._cur_moderation

    async def async_set_humidity(self, **kwargs: Any) -> None:
        """Set new humidity limit."""
        if (humidity := kwargs.get(ATTR_HUMIDITY)) is None:
            return
        self._attr_preset_mode = self._presets_inv.get(humidity, PRESET_NONE)
        self._target_hum = humidity
        await self._async_control_heating(force=True)
        self.async_write_ha_state()


    async def _async_sensor_hum_changed(self, event: Event[EventStateChangedData]) -> None:
        """Handle humidity changes."""
        new_state = event.data["new_state"]
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        self._async_update_hum(new_state)
        await self._async_control_heating()
        self.async_write_ha_state()


    @callback
    def _async_update_hum(self, state: State) -> None:
        """Update current humidity with latest state from sensor_hum."""
        try:
            cur_hum = float(state.state)
            if not math.isfinite(cur_hum) or cur_hum < 0 or cur_hum > 1:
                raise ValueError(f"SensorHum has illegal state {state.state}")  # noqa: TRY301
            self._cur_hum = cur_hum
        except ValueError as ex:
            _LOGGER.error("Unable to update from sensor: %s", ex)