# CLOUDY NOTES

## Setting up and loading the input SEDs
We want to use cloudy commands to combine SB99 models to create the input SEDs

These are then loaded into the Cloudy input object using CloudyInput.set_star()
This then reads the table.

Needed:
    Function for making linear combinations of models

## STOPPING CRITERIA
We should run for matter-bounded and density-bounded and iterate our models over this.
