# pancakeOrWaffle
Machine Learning and Google Image web scraping tool

Thus project was designed for personal use, however for those curious, to use this project and to first get it running you must:

1. Install Chromedriver
2. Install needed libraries
3. Gather a dataset (does not have to be Pancakes or Waffles)

Gathering a dataset can be done using the scraper.py tool, all that needs to be done is specifiy the search term and total images

The tool also assumes a folder structure for the images of the following:

(80% of images)

DatasetTrain
|___SearchTerm1Folder
      |___SearchTerm2Folder
      |___etc...

(20% of images)

DatasetTest
      |___SearchTerm1Folder
      |___SearchTerm2Folder
      |___etc...
 
