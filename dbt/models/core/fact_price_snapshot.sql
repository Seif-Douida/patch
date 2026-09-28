-- Daily US-store prices, going forward only (Steam has no price history endpoint).
select
    appid as game_key,
    cast(convert(char(8), snapshot_date, 112) as int) as date_key,
    ok,
    initial_price,
    final_price,
    discount_percent as discount_pct,
    currency
from {{ ref('stg_steam_price') }}
