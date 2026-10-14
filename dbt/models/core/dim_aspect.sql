select
    code as aspect_key,
    code,
    name,
    description,
    taxonomy_version
from {{ ref('aspects') }}
