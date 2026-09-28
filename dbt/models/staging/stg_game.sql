select
    appid,
    name,
    genres,
    role,
    backfill_start,
    release_date,
    developer,
    publisher
from {{ source('raw', 'game') }}
