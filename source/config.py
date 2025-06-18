import tomllib

VALID_PROVIDERS = ["yahoo_finance", "fmp", "bloomberg", "kaxanuk_dc"]


# Get Data Provider from .TOML
def get_data_provider(
        config_path=r"..\config\config.toml"
) -> str:

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    provider = config.get("settings", {}).get("provider", "").strip().lower()

    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Invalid provider '{provider}'. "
            f"Please choose one of: {', '.join(VALID_PROVIDERS)}"
        )

    return provider


# Get the API Key from .TOML
def get_api_key(
        config_path=r"..\config\config.toml",
        provider='fred',
) -> str:

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    api_key = config.get("api_keys", {}).get(provider, "").strip().lower()

    return api_key