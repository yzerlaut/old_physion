# Electrophy module

## Conversion from Igor ".pxp" files to HDF5 datafiles

1) Load and compile the  [Export-to-hdf5.ipf](Export-to-hdf5.ipf) script into the Igor software

2) In the Igor command interface run:

```
convert()
```

This will pop up a menu to select a folder. Select the folder of interest, the script will run into all subfolders of the arborescence and convert all Igor experminent files (".pxp" files) into HDF5 files ("/.h5" files).

Check the output in the command prompt,

```
  i=  0  ) generating  F:Data_Nunzio:2020:May:nm14May2020c2:nm14May2020c2_000.h5 
  i=  1  ) generating  F:Data_Nunzio:2020:May:nm14May2020c2:nm14May2020c2_001.h5 
  i=  2  ) generating  F:Data_Nunzio:2020:May:nm14May2020c2:nm14May2020c2_log0.h5
  i=  3  ) generating  F:Data_Nunzio:2020:May:nm14May2020c1:nm14May2020c1_000.h5 
  i=  4  ) generating  F:Data_Nunzio:2020:May:nm14May2020c1:nm14May2020c1_001.h5 
  i=  5  ) generating  F:Data_Nunzio:2020:May:nm14May2020c1:nm14May2020c1_log0.h5
  i=  6  ) generating  F:Data_Nunzio:2020:May:nm14May2020c0:nm14May2020c0_000.h5 
  i=  7  ) generating  F:Data_Nunzio:2020:May:nm14May2020c0:nm14May2020c0_001.h5 
  i=  8  ) generating  F:Data_Nunzio:2020:May:nm14May2020c0:nm14May2020c0_log0.h5
  i=  9  ) generating  F:Data_Nunzio:2020:May:nm13May2020c4:nm13May2020c4_000.h5 
  i=  10  ) generating  F:Data_Nunzio:2020:May:nm13May2020c4:nm13May2020c4_001.h5
  i=  11  ) generating  F:Data_Nunzio:2020:May:nm13May2020c4:nm13May2020c4_002.h5
  i=  12  ) generating  F:Data_Nunzio:2020:May:nm13May2020c4:nm13May2020c4_log0.h5
  i=  13  ) generating  F:Data_Nunzio:2020:May:nm13May2020c3:nm13May2020c3_000.h5 
  i=  14  ) generating  F:Data_Nunzio:2020:May:nm13May2020c3:nm13May2020c3_001.h5 
  [...]
```
If a given file is corrupted and stops the script, you can restart from the point where it failed with:

```
convert(i=15)
```