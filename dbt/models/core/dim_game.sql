-- game_key is Steam's appid: stable, unique, and what every other table already carries.
select
    appid as game_key,
    appid,
    name,
    genres,
    developer,
    publisher,
    release_date,
    role,
    backfill_start
from {{ ref('stg_game') }}
