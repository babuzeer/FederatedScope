"""
Attack plugin registry for the modular membership inference framework.

Usage
-----
Register a plugin (typically done in the plugin module itself)::

    from federatedscope.contrib.attack.registry import AttackRegistry
    from federatedscope.contrib.attack.base import AttackPlugin, AttackResult

    @AttackRegistry.register
    class BlackboxLoss(AttackPlugin):
        @property
        def name(self) -> str:
            return "blackbox_loss"

        def compute_scores(self, target_data, shadow_data=None,
                           global_model=None, config=None):
            member_scores = -self._get_last_round(target_data['train_losses'])
            nonmember_scores = -target_data['test_losses']
            return AttackResult(self.name, member_scores, nonmember_scores)

Retrieve and run plugins::

    plugins = AttackRegistry.get_all(gradient_available=True,
                                     cross_round_available=True)
    for plugin in plugins:
        result = plugin.compute_scores(target_data, shadow_data)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class AttackRegistry:
    """Central registry for ``AttackPlugin`` subclasses.

    All methods are class-methods so the registry is a true singleton
    without requiring explicit instantiation.

    Class Variables
    ---------------
    _plugins : dict[str, Type[AttackPlugin]]
        Maps plugin name strings to their *class* objects.
    """

    _plugins: Dict[str, type] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, plugin_class: type) -> type:
        """Register *plugin_class* and return it unchanged.

        Designed to be used as a class decorator::

            @AttackRegistry.register
            class MyAttack(AttackPlugin):
                ...

        The plugin name is read from an *instance* of the class (via the
        ``name`` property).  A temporary instance is created solely for
        this purpose; no side-effects occur on the class itself.

        Raises
        ------
        ValueError
            If a plugin with the same name is already registered.
        """
        # Instantiate temporarily to read the name property
        try:
            tmp = plugin_class()
            plugin_name = tmp.name
        except Exception as exc:
            raise RuntimeError(
                f"Failed to instantiate {plugin_class.__name__} during "
                f"registration: {exc}"
            ) from exc

        if plugin_name in cls._plugins:
            raise ValueError(
                f"AttackPlugin '{plugin_name}' is already registered "
                f"(registered by {cls._plugins[plugin_name].__name__}). "
                f"Each plugin must have a unique name."
            )

        cls._plugins[plugin_name] = plugin_class
        logger.debug(f"AttackRegistry: registered plugin '{plugin_name}' "
                     f"({plugin_class.__name__})")
        return plugin_class

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> type:
        """Return the plugin *class* registered under *name*.

        Parameters
        ----------
        name : str
            The plugin name string (as returned by ``AttackPlugin.name``).

        Returns
        -------
        Type[AttackPlugin]
            The plugin class (not an instance).

        Raises
        ------
        KeyError
            If no plugin is registered under *name*.  The error message
            lists all currently registered plugin names.
        """
        if name not in cls._plugins:
            available = ", ".join(sorted(cls._plugins.keys())) or "<none>"
            raise KeyError(
                f"No attack plugin named '{name}'. "
                f"Available plugins: [{available}]"
            )
        return cls._plugins[name]

    @classmethod
    def list_plugins(cls) -> Dict[str, Dict]:
        """Return a summary dict of all registered plugins.

        Returns
        -------
        dict[str, dict]
            Maps each plugin name to a dict with keys:
            ``class_name``, ``needs_gradient``, ``is_cross_round``,
            ``needs_shadow``.
        """
        summary = {}
        for name, plugin_class in cls._plugins.items():
            try:
                tmp = plugin_class()
                summary[name] = {
                    "class_name": plugin_class.__name__,
                    "needs_gradient": tmp.needs_gradient,
                    "is_cross_round": tmp.is_cross_round,
                    "needs_shadow": tmp.needs_shadow,
                }
            except Exception as exc:
                summary[name] = {
                    "class_name": plugin_class.__name__,
                    "error": str(exc),
                }
        return summary

    @classmethod
    def get_all(
        cls,
        gradient_available: bool = True,
        cross_round_available: bool = True,
        shadow_available: bool = True,
        names: Optional[List[str]] = None,
    ) -> List:
        """Return instantiated plugin objects filtered by data availability.

        Parameters
        ----------
        gradient_available : bool
            If ``False``, plugins that require gradient features
            (``needs_gradient=True``) are excluded.
        cross_round_available : bool
            If ``False``, plugins that require multi-round data
            (``is_cross_round=True``) are excluded.
        shadow_available : bool
            If ``False``, plugins that require shadow-client data
            (``needs_shadow=True``) are excluded.
        names : list of str, optional
            If provided, only plugins whose names are in this list are
            considered (subject to the availability filters above).

        Returns
        -------
        list[AttackPlugin]
            Instantiated plugin objects in registration order.
        """
        result = []
        candidate_names = names if names is not None else list(cls._plugins.keys())

        for name in candidate_names:
            if name not in cls._plugins:
                logger.warning(f"AttackRegistry.get_all: unknown plugin '{name}', skipping")
                continue

            plugin_class = cls._plugins[name]
            try:
                instance = plugin_class()
            except Exception as exc:
                logger.error(f"AttackRegistry: failed to instantiate '{name}': {exc}")
                continue

            # Apply availability filters
            if instance.needs_gradient and not gradient_available:
                logger.debug(f"Skipping '{name}': gradient features not available")
                continue
            if instance.is_cross_round and not cross_round_available:
                logger.debug(f"Skipping '{name}': cross-round data not available")
                continue
            if instance.needs_shadow and not shadow_available:
                logger.debug(f"Skipping '{name}': shadow data not available")
                continue

            result.append(instance)

        logger.info(
            f"AttackRegistry.get_all: returning {len(result)}/{len(candidate_names)} "
            f"plugins (gradient={gradient_available}, "
            f"cross_round={cross_round_available}, "
            f"shadow={shadow_available})"
        )
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @classmethod
    def clear(cls) -> None:
        """Remove all registered plugins.  Mainly useful in unit tests."""
        cls._plugins.clear()

    @classmethod
    def __repr__(cls) -> str:
        names = sorted(cls._plugins.keys())
        return f"AttackRegistry(plugins={names})"
