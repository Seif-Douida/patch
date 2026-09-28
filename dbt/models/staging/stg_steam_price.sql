select
    appid,
    snapshot_date,
    ok,
    currency,
    initial_price,
    final_price,
    discount_percent
from {{ source('raw', 'steam_price') }}
