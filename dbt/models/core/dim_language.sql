select
    code as language_key,
    code,
    name
from {{ ref('languages') }}
