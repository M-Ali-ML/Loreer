#!/bin/bash

cd "$(dirname "$0")"

# the link can be found here 
# https://leagueoflegends.fandom.com/wiki/Special:Statistics
# you will find the dumbs to any wiki in the Special:Statistics page

curl -O https://s3.amazonaws.com/wikia_xml_dumps/l/le/leagueoflegends_pages_current.xml.7z

7za x leagueoflegends_pages_current.xml.7z