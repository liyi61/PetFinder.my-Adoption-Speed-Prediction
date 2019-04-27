---
layout: page
title: Exploratory Data Analysis
bigimg: /img/main5.png
---

<div class='tableauPlaceholder' id='viz1556060748372' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;pe&#47;pet1&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='pet1&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;pe&#47;pet1&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div> 
<script type='text/javascript'>var divElement = document.getElementById('viz1556060748372');var vizElement = divElement.getElementsByTagName('object')[0];vizElement.style.width='1000px';vizElement.style.height='1027px';                    var scriptElement = document.createElement('script');scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js'; vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>


<br>
<br>
## The Impact of Pet Type for Adoption Speed

![barchart](/img/Screen Shot 2019-04-25 at 4.38.22 PM.png)

According to the previous graph, the base rate of pets with Adoption speed 0 is 410 / 14993 = __0.027__;  
There are 6861 cats in train dataset and 240 of them have Adoption Speed 0. So the rate is 240 / 6861 = __0.035__;  
Above diagram shows the differences between the general adoption rate and type-specified adoption rate. We can see that cats are more likely to be adopted early than dogs. On the other hand more dogs are adopted at slower speed.  

<br>
## About Pet Names

"Name" is the only column that has missing values in the data.   
__Rate of unnamed pets in train data: 8.4173%.__   
__Rate of unnamed pets in test data: 7.6748%.__  

![wordcloud](/img/Screen Shot 2019-04-25 at 7.50.40 PM.png)


![noname](/img/Screen Shot 2019-04-25 at 10.25.28 PM.png)
We can see that pets with no name have lower chance being adopted.  

## Age VS Adoption Speed

This is a graph showing the distribution of pets' age. Most of the pets are still at their very early stage of life.   

<div class='tableauPlaceholder' id='viz1556249115180' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ag&#47;age_15562491727790&#47;Sheet4&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='age_15562491727790&#47;Sheet4' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ag&#47;age_15562491727790&#47;Sheet4&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div> 
<script type='text/javascript'>var divElement = document.getElementById('viz1556249115180');var vizElement = divElement.getElementsByTagName('object')[0];vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>

<br>

![age](/img/age.png)

<br>

## Adoption Speed Trend by Age

We can see that young pets are adopted relatively fast and the peek is at two-month old.  

<div>
    <a href="https://plot.ly/~liyi0907/5/?share_key=Gu3FmSC7r4vfZnfX5ui92v" target="_blank" title="Plot 5" style="display: block; text-align: center;"><img src="https://plot.ly/~liyi0907/5.png?share_key=Gu3FmSC7r4vfZnfX5ui92v" alt="Plot 5" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
    <script data-plotly="liyi0907:5" sharekey-plotly="Gu3FmSC7r4vfZnfX5ui92v" src="https://plot.ly/embed.js" async></script>
</div>


## The Impact of Breeds


Rate of pure breed pets in train data: __71.7802%__.    
Rate of pure breed pets in test data: __77.9635%__.    
The breed information looks a bit messy and confusing. These numbers exist under the assumption that pets without "breed2" are pure breed. That is incorrect. Let's take a look at the top 15 combination of "breed1" and "breed2":  


| Breed Name | Count |
| :------: |:---: |
| Mixed_Breed__- | 5573 |
| Domestic_Short_Hair__- | 4042 |
| Domestic_Medium_Hair__- | 1264 |
| Mixed_Breed__Mixed_Breed | 1188 |
| Tabby__- | 379 |
| Domestic_Short_Hair__Domestic_Short_Hair | 320 |
| Domestic_Long_Hair__- | 244 |
| Shih_Tzu__- | 204 |
| Poodle__- | 153 |
| Siamese__- | 152 |
| Labrador_Retriever__Mixed_Breed | 132 |
| Golden_Retriever__- | 123 |
| Domestic_Medium_Hair__Domestic_Medium_Hair | 110 |
| Domestic_Medium_Hair__Domestic_Short_Hair | 104 |
| Calico__- | 101 |

The assumption is wrong because a lot of people put "Mixed Breed" for breed1 and left breed2 blank. And all cats starting with "Domestic" seem not pure breed cats. After specifying all theses "Domestic" cats and mixed breed dogs as non-pure breed pets, it comes to the conclusion that pure breed pets are doing better regarding the adoption speed.  

```python
for i, row in all_data[all_data['Breed1_name'].str.contains('air')].iterrows():
    if 'Short' in row['Breed1_name'] and row['FurLength'] == 1:
        pass
    elif 'Medium' in row['Breed1_name'] and row['FurLength'] == 2:
        pass
    elif 'Long' in row['Breed1_name'] and row['FurLength'] == 3:
        pass
    else:
        c += 1
        strange_pets.append((row['PetID'], row['Breed1_name'], row['FurLength']))
```

Another confusing feature is fur-length. There are __964__ pets whose breed and fur length don't match.   
Here are some examples:  

![furlength](/img/Screen Shot 2019-04-26 at 11.25.27 PM.png)

Sometiems breed is more accurate, sometime fur length is. One thing could be done is creatign a new feautre about this mismatch situation. 


## The Impact of Gender on Adoption Speed

__1 = Male, 2 = Female, 3 = Mixed__  
*Gender is "Mixed", if profile represents group of pets. 



<div class='tableauPlaceholder' id='viz1556328946904' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ge&#47;gender_15563290507740&#47;Sheet5&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='gender_15563290507740&#47;Sheet5' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ge&#47;gender_15563290507740&#47;Sheet5&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                
<script type='text/javascript'>var divElement = document.getElementById('viz1556328946904');var vizElement = divElement.getElementsByTagName('object')[0];vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';var scriptElement = document.createElement('script');scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>

<br>
![genderdist](/img/Screen Shot 2019-04-26 at 9.42.29 PM.png)

According to this graph, we see that the number of Female pets is significantly larger than the number of Male pets in the training set, and that the Male pets are adopted quicker. Groups are also adopted slower than Male pets. 


## The Influence of Health Status

In the data, there are four variables that are related to the health of a pet or the medical actions that a pet has had. These are:  
* Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
* Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
* Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
* Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)

This could be a very important feature because most people would surely prefer adopting a healthy pet. Also, as there are costs associated with vaccinating, deworming, and steralizing, people might be willing to adopt pets that have had these medical actions.  

<div class='tableauPlaceholder' id='viz1556393988694' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;he&#47;health1_15563940542940&#47;Sheet7&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='health1_15563940542940&#47;Sheet7' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;he&#47;health1_15563940542940&#47;Sheet7&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                
<script type='text/javascript'>var divElement = document.getElementById('viz1556393988694');var vizElement = divElement.getElementsByTagName('object')[0];vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>











