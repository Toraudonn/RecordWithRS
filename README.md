# RecordWithRS

Record RGB and Depths data using Intel RealSense sensor

## How-To

### Make!

- Change directory to the project root.
- Run `./scripts/build.sh` (you may need to `chmod` it).
- Run `./scripts/rm_data.sh`.
    - This script removes the data directory if you already had it, then creates a fresh data directory.
- Run `./build/rs-capture`.

### Data Directory

The `script/rm_data.sh` script will remove all of the data in the data directory.
Therefore you shouldn't run this script if you don't want to delete your data.

## TODOs

- [x] Create the data saving script
 
