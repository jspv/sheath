from sheath.config import SheathConfig, SearchConfig


def test_search_config_defaults():
    cfg = SheathConfig()
    assert isinstance(cfg.search, SearchConfig)
    assert cfg.search.provider == "tavily"
    assert cfg.search.api_key is None
    assert cfg.search.max_results == 5
    assert cfg.search.timeout_s == 20.0


def test_search_config_independent_between_instances():
    assert SheathConfig().search is not SheathConfig().search
