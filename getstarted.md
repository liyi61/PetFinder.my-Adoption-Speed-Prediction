---
layout: page
title: Data Overview
subtitle: 
bigimg: /img/main4.jpg
---

The datasets are provided by Kaggle and the data included text, tabular, and image data. First, Let's look at the preview of the traning data:


![preview](/img/Screen Shot 2019-04-22 at 3.46.49 AM.png)



## Data Fields

*__PetID__ - Unique hash ID of pet profile
*__AdoptionSpeed__ - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
*__Type__ - Type of animal (1 = Dog, 2 = Cat)
*__Name__ - Name of pet (Empty if not named)
*__Age__ - Age of pet when listed, in months
*__Breed1__ - Primary breed of pet (Refer to BreedLabels dictionary)
*__Breed2__ - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
*__Gender__ - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
*__Color1__ - Color 1 of pet (Refer to ColorLabels dictionary)
*__Color2__ - Color 2 of pet (Refer to ColorLabels dictionary)
*__Color3__ - Color 3 of pet (Refer to ColorLabels dictionary)
*__MaturitySize__ - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
*__FurLength__ - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
*__Vaccinated__ - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
*__Dewormed__ - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
*__Sterilized__ - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
*__Health__ - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
*__Quantity__ - Number of pets represented in profile
*__Fee__ - Adoption fee (0 = Free)
*__State__ - State location in Malaysia (Refer to StateLabels dictionary)
*__RescuerID__ - Unique hash ID of rescuer
*__VideoAmt__ - Total uploaded videos for this pet
*__PhotoAmt__ - Total uploaded photos for this pet
*__Description__ - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.


### 1. Fork the Beautiful Jekyll repository 

Fork the [repository](https://github.com/daattali/beautiful-jekyll) by clicking the Fork button on the top right corner in GitHub.

### 2. Rename the repository to `yourusername.github.io`

This will create a GitHub User page ready with the **Beautiful Jekyll** template that will be available at http://yourusername.github.io (sometimes it takes a few minutes).

### 3. Customize settings

Edit the `_config.yml` file to change all the settings to reflect your site.  The settings in the file are fairly self-explanatory and I added comments inside the file to help you further.  Every time you make a change to any file, your website will get rebuilt and should be updated at `yourusername.github.io` within a minute.

You can now visit your shiny new website, which will be seeded with several sample blog posts and a couple other pages.

---

See how easy that is? I wasn't lying - it really can be done in two minutes.

<div class="get-started-wrap">
  <a class="btn btn-success btn-lg get-started-btn" href="https://github.com/daattali/beautiful-jekyll#readme">Get Started!</a>
</div>
