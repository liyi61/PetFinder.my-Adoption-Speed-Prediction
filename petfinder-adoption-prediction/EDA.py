import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from PIL import Image
from wordcloud import WordCloud
import plotly.offline as py
import plotly.graph_objs as go
import random
import warnings
warnings.filterwarnings("ignore")
pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)
import os
from scipy.misc import imread

breeds = pd.read_csv('./breed_labels.csv')
colors = pd.read_csv('./color_labels.csv')
states = pd.read_csv('./state_labels.csv')

train = pd.read_csv('./train/train.csv')
test = pd.read_csv('./test/test.csv', engine='python')
sub = pd.read_csv('./test/sample_submission.csv')

train['dataset_type'] = 'train'
test['dataset_type'] = 'test'
all_data = pd.concat([train, test])
all_data['Type'] = all_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
images = [i.split('-')[0] for i in os.listdir('./train_images/')]

# ****************************************** EDA *************************************************
# Number of animals in different adoption speed classes
fig1 = plt.figure()
train['AdoptionSpeed'].value_counts().sort_index().plot('bar', color='orange')
plt.title('The classes counts distribution for different AdoptionSpeed')
# plt.show()
fig1.savefig('./result_image/1_AdoptionSpeed_count.jpg')

# # Adoption speed classes ratios
fig2 = plt.figure()
g = sns.countplot(x='AdoptionSpeed', data=all_data.loc[all_data['dataset_type'] == 'train'],  palette="Set2")
plt.title('The classes rates distribution for different AdoptionSpeed')
ax = g.axes
for p in ax.patches:
     ax.annotate(f"{p.get_height() * 100 / train.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),
         textcoords='offset points')
plt.show()
fig2.savefig('./result_image/2_AdoptionSpeed_rate.jpg')

# Showing number of different types：1-Dog；2-Cat
fig3 = plt.figure()
# all_data['Type'] = all_data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
sns.countplot(x='dataset_type', data=all_data, hue='Type', palette="Set2")
plt.title('The number of cats and dogs in the given data')
main_count = train['AdoptionSpeed'].value_counts(normalize=True).sort_index()
plt.show()
fig3.savefig('./result_image/3_number_of_dog_cat.jpg')

# Adoption Speed Comparison beween dogs and cats
def prepare_plot_dict(df, col, main_count):
    main_count = dict(main_count)
    plot_dict = {}
    for i in df[col].unique():
        val_count = dict(df.loc[df[col] == i, 'AdoptionSpeed'].value_counts().sort_index())
        for k, v in main_count.items():
            if k in val_count:
                plot_dict[val_count[k]] = ((val_count[k] / sum(val_count.values())) / main_count[k]) * 100 - 100
            else:
                plot_dict[0] = 0
    return plot_dict
main_count = train['AdoptionSpeed'].value_counts(normalize=True).sort_index()
def make_count_plot(df, x, hue='AdoptionSpeed', title='', main_count=main_count ):
    g = sns.countplot(x=x, data=df, hue=hue, palette="Set2")
    plt.title(title)
    ax = g.axes
    plot_dict = prepare_plot_dict(df, x, main_count)
    for p in ax.patches:
        h = p.get_height() if str(p.get_height()) != 'nan' else 0
        text = f"{plot_dict[h]:.0f}%" if plot_dict[h] < 0 else f"+{plot_dict[h]:.0f}%"
        ax.annotate(text, (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='center', fontsize=11, color='green' if plot_dict[h] > 0 else 'red', rotation=0,
                    xytext=(0, 10),
                    textcoords='offset points')
fig4 = plt.figure()
make_count_plot(df=all_data.loc[all_data['dataset_type'] == 'train'], x='Type', title='The impact of pet type for AdoptionSpeed')
plt.show()
fig4.savefig('./result_image/4_impact_of_pet_type.jpg')

# Name vs. Adoption Speed
fig5 = plt.figure()
text_dog = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Name'].fillna('').values)
dog_mask = imread("./test_images/4a288da21-3.jpg")
wordcloud = WordCloud(max_font_size=150, background_color='white',
                      width=1000, height=1000, mask=dog_mask, colormap='Dark2').generate(text_dog)
plt.imshow(wordcloud)
plt.title('Top dog names')
plt.axis("off")
fig5.savefig('./result_image/5_Dog_name.jpg')

fig6 = plt.figure()
text_cat = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Name'].fillna('').values)
cat_mask = imread("./test_images/0b1d74132-2.jpg")
wordcloud = WordCloud(max_font_size=150, background_color='white',
                      width=1000, height=1000, mask=cat_mask, colormap='Dark2').generate(text_cat)
plt.imshow(wordcloud)
plt.title('Top cat names')
plt.axis("off")

plt.show()
fig6.savefig('./result_image/6_Cat_name.jpg')

# The most popular names vs Adoption Speed
print('-----------------The most popular names vs Adoption Speed---------------------')
for n in train['Name'].value_counts().index[:5]:
    print(n)
    print(train.loc[train['Name'] == n, 'AdoptionSpeed'].value_counts().sort_index())
    print('')

# # the ratio of No_Name pets in train and test sets
print('-------------------the ratio of No_Name pets in train and test sets-------------------')
train['Name'] = train['Name'].fillna('Unnamed')
test['Name'] = test['Name'].fillna('Unnamed')
all_data['Name'] = all_data['Name'].fillna('Unnamed')

train['No_name'] = 0
train.loc[train['Name'] == 'Unnamed', 'No_name'] = 1
test['No_name'] = 0
test.loc[test['Name'] == 'Unnamed', 'No_name'] = 1
all_data['No_name'] = 0
all_data.loc[all_data['Name'] == 'Unnamed', 'No_name'] = 1

print(f"Rate of unnamed pets in train data: {train['No_name'].sum() * 100 / train['No_name'].shape[0]:.4f}%.")
print(f"Rate of unnamed pets in test data: {test['No_name'].sum() * 100 / test['No_name'].shape[0]:.4f}%.")

# # In No name pets, ratio of dogs and cats in different classes of adoption speed
print(pd.crosstab(train['No_name'], train['AdoptionSpeed'], normalize='index'))

# percentage of no name pets in the data
fig7 = plt.figure()
make_count_plot(df=all_data.loc[all_data['dataset_type'] == 'train'], x='No_name',
                title='The AdoptionSpeed for unnamed pets')
fig7.savefig('./result_image/7_AdoptionSpeed_unnamed_pets.jpg')

# #the distribution of age in train and test sets
fig8 = plt.figure()
plt.title('The pets age distribution for train and test data')
train['Age'].plot('hist', label='train', color='magenta')
test['Age'].plot('hist', label='test', color='blue')
plt.xlim((0, 160))
plt.xlabel('Age')
plt.legend()
fig8.savefig('./result_image/8_age_distribution.jpg')

# # Trend of age
print(train['Age'].value_counts().head(10))
#
# # age vs adoption speed
fig9 = plt.figure()
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Type", data=train, palette="muted", split=True)
plt.title('The impact of pet age for AdoptionSpeed')
fig9.savefig('./result_image/9_impact_of_pet_age.jpg')

# # The impace age has on adoption speed 
data = []
for a in range(5):
    df = train.loc[train['AdoptionSpeed'] == a]

    data.append(go.Scatter(
        x=df['Age'].value_counts().sort_index().index,
        y=df['Age'].value_counts().sort_index().values,
        name=str(a)
    ))

layout = go.Layout(dict(title="AdoptionSpeed trends along pets age changing",
                        xaxis=dict(title='Age (months)'),
                        yaxis=dict(title='Counts'),
                        )
                   )
py.plot(dict(data=data, layout=layout), filename='basic-line')


# # Impact of breed
train['Pure_breed'] = 0
train.loc[train['Breed2'] == 0, 'Pure_breed'] = 1
test['Pure_breed'] = 0
test.loc[test['Breed2'] == 0, 'Pure_breed'] = 1
all_data['Pure_breed'] = 0
all_data.loc[all_data['Breed2'] == 0, 'Pure_breed'] = 1
print(f"Rate of pure breed pets in train data: {train['Pure_breed'].sum() * 100 / train['Pure_breed'].shape[0]:.4f}%.")
print(f"Rate of pure breed pets in test data: {test['Pure_breed'].sum() * 100 / test['Pure_breed'].shape[0]:.4f}%.")


fig11 = plt.figure()
sns.countplot(x='dataset_type', data=all_data, hue='Pure_breed', palette="Set2")
plt.title('The number of pure/not-pure breed  pets')
fig11.savefig('./result_image/11_Number_of_pure_nopure_breed_pets.jpg')

fig12 = plt.figure()
make_count_plot(train.loc[train['Type'] == 1], x="Pure_breed", title='The AdoptionSpeed of pure breed dogs')
fig12.savefig('./result_image/12_AdoptionSpeed_of_pure_breed_dogs.jpg')

fig13 = plt.figure()
make_count_plot(train.loc[train['Type'] == 2], x="Pure_breed", title='The AdoptionSpeed of pure breed cats')
fig13.savefig('./result_image/13_AdoptionSpeed_of_pure_breed_cats.jpg')

# # # Word cloud of animals' breed
breeds_dict = {k: v for k, v in zip(breeds['BreedID'], breeds['BreedName'])}
all_data['Breed1_name'] = all_data['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'Unknown')
all_data['Breed2_name'] = all_data['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else '-')

fig14 = plt.figure()
text_dog1 = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=150, width=1000, height=1000, colormap='Dark2').generate(text_dog1)
plt.imshow(wordcloud)
plt.title('Top dog breed1')
plt.axis("off")
fig14.savefig('./result_image/14_dog_breed1_wordcloud.jpg')

fig15 = plt.figure()
text_cat1 = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Breed1_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=150, width=1000, height=1000).generate(text_cat1)
plt.imshow(wordcloud)
plt.title('Top cat breed1')
plt.axis("off")
fig15.savefig('./result_image/15_cat_breed1_wordcloud.jpg')

fig16 = plt.figure()
text_dog2 = ' '.join(all_data.loc[all_data['Type'] == 'Dog', 'Breed2_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=150, width=1000, height=1000, colormap='Dark2').generate(text_dog2)
plt.imshow(wordcloud)
plt.title('Top dog breed2')
plt.axis("off")
fig16.savefig('./result_image/16_dog_breed2_wordcloud.jpg')

fig17 = plt.figure()
text_cat2 = ' '.join(all_data.loc[all_data['Type'] == 'Cat', 'Breed2_name'].fillna('').values)
wordcloud = WordCloud(max_font_size=150, width=1000, height=1000).generate(text_cat2)
plt.imshow(wordcloud)
plt.title('Top cat breed2')
plt.axis("off")
fig17.savefig('./result_image/17_cat_breed2_wordcloud.jpg')
# plt.show()

# # Combination of breed1 and breed2
print((all_data['Breed1_name'] + '__' + all_data['Breed2_name']).value_counts().head(15))

# # Gender Study
# # 1 = Male, 2 = Female, 3 = Mixed
# the number of female and males pets in train and test sets
fig18 = plt.figure()
sns.countplot(x='dataset_type', data=all_data, hue='Gender', palette="Set2")
plt.title('The number of different genders pets in the given data')
fig18.savefig('./result_image/18_Number_different_gender.jpg')


fig19 = plt.figure()
make_count_plot(df=train, x='Gender', title='The impact of pet gender for AdoptionSpeed')
fig19.savefig('./result_image/19_impact_of_pet_gender.jpg')


# # Color
# # Distribution of colors 
colors_dict = {k: v for k, v in zip(colors['ColorID'], colors['ColorName'])}
train['Color1_name'] = train['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color2_name'] = train['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
train['Color3_name'] = train['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

test['Color1_name'] = test['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color2_name'] = test['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
test['Color3_name'] = test['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

all_data['Color1_name'] = all_data['Color1'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
all_data['Color2_name'] = all_data['Color2'].apply(lambda x: colors_dict[x] if x in colors_dict else '')
all_data['Color3_name'] = all_data['Color3'].apply(lambda x: colors_dict[x] if x in colors_dict else '')

def make_factor_plot(df, x, col, title, main_count=main_count, hue=None, ann=True, col_wrap=3):
    if hue:
        g = sns.factorplot(col, col=x, data=df, kind='count', col_wrap=col_wrap, hue=hue, palette="Set2")
    else:
        g = sns.factorplot(col, col=x, data=df, kind='count', col_wrap=col_wrap, palette="Set2")
    plt.subplots_adjust(top=0.9)
    plt.suptitle(title, y=1.05)
    ax = g.axes
    plot_dict = prepare_plot_dict(df, x, main_count)
    if ann:
        for a in ax:
            for p in a.patches:
                text = f"{plot_dict[p.get_height()]:.0f}%" if plot_dict[p.get_height()] < 0 else f"+{plot_dict[p.get_height()]:.0f}%"
                a.annotate(text, (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='green' if plot_dict[p.get_height()] > 0 else 'red', rotation=0, xytext=(0, 10),
                     textcoords='offset points')
                return g
fig20 = sns.factorplot('dataset_type', col='Type', data=all_data, kind='count', hue='Color1_name', palette=['black', 'peru', 'peachpuff', 'Gray', 'gold', 'white', 'yellow'])
plt.subplots_adjust(top=0.85)
sns.set_style("dark")
plt.suptitle('The number of different color pets in the given data')
fig20.savefig('./result_image/20_Number_of_different_color.jpg')



train['full_color'] = (train['Color1_name'] + '__' + train['Color2_name'] + '__' + train['Color3_name']).str.replace('__', '')
test['full_color'] = (test['Color1_name'] + '__' + test['Color2_name'] + '__' + test['Color3_name']).str.replace('__', '')
all_data['full_color'] = (all_data['Color1_name'] + '__' + all_data['Color2_name'] + '__' + all_data['Color3_name']).str.replace('__', '')

fig21 = make_factor_plot(df=train.loc[train['full_color'].isin(list(train['full_color'].value_counts().index)[:12])],
                         x='full_color', col='AdoptionSpeed',
                         title='The number of different color pets and AdoptionSpeed')
fig21.savefig('./result_image/21_Number_of_different_color_AdoptionSpeed.jpg')

# # Main colors for female and male animals 
print('-----------------Main colors for female and male animals ---------------------')
gender_dict = {1: 'Male', 2: 'Female', 3: 'Mixed'}
for i in all_data['Type'].unique():
    for j in all_data['Gender'].unique():
        df = all_data.loc[(all_data['Type'] == i) & (all_data['Gender'] == j)]
        top_colors = list(df['full_color'].value_counts().index)[:5]
        j = gender_dict[j]
        print(f"Most popular colors of {j} {i}s: {' '.join(top_colors)}")

# # Maturity size 
# # maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)

fig22 = plt.figure()
sns.countplot(x='dataset_type', data=all_data, hue='MaturitySize', palette="Set2")
plt.title('The number of pets with different MaturitySize')
fig22.savefig('./result_image/22_number_with_different_MaturitySize.jpg')

fig23 = plt.figure(figsize=(15, 10))
make_count_plot(df=train, x='MaturitySize', title='The number of different MaturitySize pets and AdoptionSpeed')
fig23.savefig('./result_image/23_impact_of_pet_MaturitySize.jpg')

# # Main breeds for different maturity size 
# # Show some pictures as exmaples 
size_dict = {1: 'Small', 2: 'Medium', 3: 'Large', 4: 'Extra Large'}
for t in all_data['Type'].unique():
    for m in all_data['MaturitySize'].unique():
        df = all_data.loc[(all_data['Type'] == t) & (all_data['MaturitySize'] == m)]
        top_breeds = list(df['Breed1_name'].value_counts().index)[:3]
        m = size_dict[m]
        print(f"Most common Breeds of {m} {t}s:")

        fig = plt.figure(figsize=(15, 8))
        plt.title(f"Most common Breeds of {m} {t}s", fontsize='xx-large', fontweight='bold', y=1.05)
        # plt.subplots_adjust(top=1)
        for i, breed in enumerate(top_breeds):
            b_df = df.loc[(df['Breed1_name'] == breed) & (df['PetID'].isin(images)), 'PetID']
            if len(b_df) > 1:
                pet_id = b_df.values[1]
            else:
                pet_id = b_df.values[0]
            ax = fig.add_subplot(1, 3, i + 1, xticks=[], yticks=[])

            im = Image.open("./train_images/" + pet_id + '-1.jpg')
            plt.imshow(im)
            ax.set_title(f'Breed: {breed}')
            plt.axis("off")
        plt.show()
        fig.savefig(f'./result_image/24_Most_common_Breeds_of_{m}_{t}s')


# # Impact of FurLength
# # (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
fig25 = plt.figure()
sns.countplot(x='dataset_type', data=all_data, hue='FurLength', palette="Set2")
plt.title('The number of pets with different FurLength')
fig25.savefig('./result_image/25_number_with_different_FurLength.jpg')

fig26 = plt.figure()
make_count_plot(df=train, x='FurLength', title='The number of different FurLength pets and AdoptionSpeed')
fig26.savefig('./result_image/26_impact_of_pet_FurLength.jpg')

fig27 = plt.figure()
make_count_plot(df=train.loc[train['Type'] == 1], x='FurLength',
                    title='The number of different FurLength dogs and AdoptionSpeed')
fig27.savefig('./result_image/27_impact_of_dog_FurLength.jpg')

fig28= plt.figure()
make_count_plot(df=train.loc[train['Type'] == 2], x='FurLength',
                    title='The number of different FurLength cats and AdoptionSpeed')
fig28.savefig('./result_image/28_impact_of_cat_FurLength.jpg')

# # There are 964 data points with mismatched fur length and breed
c = 0
strange_pets = []
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

print(f"There are {c} pets whose breed and fur length don't match")
strange_pets = [p for p in strange_pets if p[0] in images]
fig29 = plt.figure(figsize=(12, 10))
plt.title('Some mismatched examples of breed and FurLength information', fontsize='xx-large', fontweight='bold',
          y=1.08)
fur_dict = {1: 'Short', 2: 'Medium', 3: 'long'}
for i, s in enumerate(random.sample(strange_pets, 9)):
    ax = fig29.add_subplot(3, 3, i+1, xticks=[], yticks=[])

    im = Image.open("./train_images/" + s[0] + '-1.jpg')
    plt.imshow(im)
    ax.set_title(f'Breed: {s[1]} \n Fur length: {fur_dict[s[2]]}')
fig29.savefig('./result_image/29_mismatched_examples_of_breed_FurLength.jpg')
plt.show()

# # Health
# # Related features are Vaccinated, Dewormed, Sterilized Health
fig30 = plt.figure()
make_count_plot(df=train, x='Vaccinated', title='Vaccinated')
plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure'])
plt.title('The impact of pet Vaccinated for AdoptionSpeed')
fig30.savefig('./result_image/30_impact_of_pet_Vaccinated.jpg')

fig31 = plt.figure()
make_count_plot(df=train, x='Dewormed', title='Dewormed')
plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure'])
plt.title('The impact of pet Dewormed for AdoptionSpeed')
fig31.savefig('./result_image/31_impact_of_pet_Dewormed.jpg')

fig32 = plt.figure()
make_count_plot(df=train, x='Sterilized', title='Sterilized')
plt.xticks([0, 1, 2], ['Yes', 'No', 'Not sure'])
plt.title('The impact of pet Sterilized for AdoptionSpeed')
fig32.savefig('./result_image/32_impact_of_pet_Sterilized.jpg')

fig33 = plt.figure()
make_count_plot(df=train, x='Health', title='Health')
plt.xticks([0, 1, 2], ['Healthy', 'Minor Injury', 'Serious Injury'])
plt.title('The impact of pet health for AdoptionSpeed')
fig33.savefig('./result_image/33_impact_of_pet_health.jpg')

# # Helath and Age 
fig34 = plt.figure()
sns.violinplot(x="AdoptionSpeed", y="Age", data=train, palette=['red', 'blue', 'yellow', 'orange', 'grey'])
plt.title('Age distribution of different AdoptionSpeed')
fig34.savefig('./result_image/34_Age_distribution_different_AdoptionSpeed.jpg')

fig35 = plt.figure()
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Vaccinated", data=train,
               palette=['red', 'blue', 'orange'])
plt.title('Age distribution of different AdoptionSpeed and Vaccinated')
fig35.savefig('./result_image/35_Age_distribution_different_AdoptionSpeed_Vaccinated.jpg')

fig36 = plt.figure()
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Dewormed", data=train,
               palette=['red', 'blue', 'orange'])
plt.title('Age distribution of different AdoptionSpeed and Dewormed')
fig36.savefig('./result_image/36_Age_distribution_different_AdoptionSpeed_Dewormed.jpg')

fig37 = plt.figure()
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Sterilized", data=train,
               palette=['red', 'blue', 'orange'])
plt.title('Age distribution of different AdoptionSpeed and Sterilized')
fig37.savefig('./result_image/37_Age_distribution_different_AdoptionSpeed_Sterilized.jpg')

fig38 = plt.figure()
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Health", data=train,
               palette=['red', 'blue', 'orange'])
plt.title('Age distribution of different AdoptionSpeed and Health')
fig38.savefig('./result_image/38_Age_distribution_different_AdoptionSpeed_Health.jpg')

# # Influence of Adoption Fee

train['Free'] = train['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
test['Free'] = test['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
all_data['Free'] = all_data['Fee'].apply(lambda x: 'Free' if x == 0 else 'Not Free')
fig39 = plt.figure()
sns.countplot(x='dataset_type', data=all_data, hue='Free', palette="Set2")
plt.title('The number of pets with Fee and Not Free in the given data')
fig39.savefig('./result_image/39_Number_different_Fee.jpg')

fig40 = plt.figure()
make_count_plot(df=train, x='Free', title='The number of different Fee pets and AdoptionSpeed')
fig40.savefig('./result_image/40_impact_of_pet_Fee.jpg')


fig41 = plt.figure()
make_count_plot(df=train.loc[train['Type'] == 1], x='Free', title='The number of different Fee dogs and AdoptionSpeed')
fig41.savefig('./result_image/41_impact_of_dog_Fee.jpg')

fig42 = plt.figure()
make_count_plot(df=train.loc[train['Type'] == 2], x='Free', title='The number of different Fee cats and AdoptionSpeed')
fig42.savefig('./result_image/42_impact_of_cat_Fee.jpg')


fig43 = plt.figure()
sns.violinplot(x="AdoptionSpeed", y="Fee", hue="Type", data=train, palette=['red', 'blue'], split=True)
plt.title('The impact of Type and Fee for AdoptionSpeed')
fig43.savefig('./result_image/43_impact_of_Type_Fee.jpg')

# # States
states_dict = {k: v for k, v in zip(states['StateID'], states['StateName'])}
train['State_name'] = train['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')
all_data['State_name'] = all_data['State'].apply(lambda x: '_'.join(states_dict[x].split()) if x in states_dict else 'Unknown')
print(all_data['State_name'].value_counts(normalize=True).head())

fig44 = make_factor_plot(df=train.loc[train['State_name'].isin(list(train.State_name.value_counts().index[:3]))],
                         x='State_name', col='AdoptionSpeed', title='The number of different states and AdoptionSpeed')
fig44.savefig('./result_image/44_impact_of_state.jpg')

# # ResecueriD
print(all_data['RescuerID'].value_counts().head())

fig45 = make_factor_plot(df=train.loc[train['RescuerID'].isin(list(train.RescuerID.value_counts().index[:5]))],
                         x='RescuerID', col='AdoptionSpeed',
                         title='The number of different rescuers and AdoptionSpeed', col_wrap=3)
fig45.savefig('./result_image/45_impact_of_Rescuer.jpg')

# # VideoAmt Analysis

print(train['VideoAmt'].value_counts())

# #
# # PhotoAmt Analysis
print(F'Maximum amount of photos in {train["PhotoAmt"].max()}')
print(train['PhotoAmt'].value_counts().head())

fig46 = make_factor_plot(df=train.loc[train['PhotoAmt'].isin(list(train.PhotoAmt.value_counts().index[:5]))],
                         x='PhotoAmt', col='AdoptionSpeed',
                         title='The number of different PhotoAmt and Adoption Speed', col_wrap=3)
fig46.savefig('./result_image/46_impact_of_PhotoAmt.jpg')

fig47 = plt.figure()
plt.hist(train['PhotoAmt'], color='deepskyblue')
plt.title('Distribution of PhotoAmt')
fig47.savefig('./result_image/47_Distribution_of_PhotoAmt.jpg')

# # Description Length Analysis

train['Description'] = train['Description'].fillna('')
train['desc_length'] = train['Description'].apply(lambda x: len(x))
fig48 = plt.figure(figsize=(16, 6))
plt.plot()
sns.violinplot(x="AdoptionSpeed", y="desc_length", hue="Type", data=train, palette=['red', 'blue'], split=True)
plt.title('The impact of Type and description length')
fig48.savefig('./result_image/48_Impact_of_desc_length.jpg')

# # Sentiment Analysis
sentiment_dict = {}
for filename in os.listdir('./train_sentiment/'):
    with open('./train_sentiment/' + filename, 'rb') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}
    sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['language'] = sentiment['language']

for filename in os.listdir('./test_sentiment/'):
    with open('./test_sentiment/' + filename, 'rb') as f:
        sentiment = json.load(f)
    pet_id = filename.split('.')[0]
    sentiment_dict[pet_id] = {}
    sentiment_dict[pet_id]['magnitude'] = sentiment['documentSentiment']['magnitude']
    sentiment_dict[pet_id]['score'] = sentiment['documentSentiment']['score']
    sentiment_dict[pet_id]['language'] = sentiment['language']

train['language'] = train['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
train['magnitude'] = train['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
train['score'] = train['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

test['language'] = test['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
test['magnitude'] = test['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
test['score'] = test['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

all_data['language'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['language'] if x in sentiment_dict else 'no')
all_data['magnitude'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['magnitude'] if x in sentiment_dict else 0)
all_data['score'] = all_data['PetID'].apply(lambda x: sentiment_dict[x]['score'] if x in sentiment_dict else 0)

fig49 = plt.figure()
sns.countplot(x='dataset_type', data=all_data, hue='language', palette="Set2")
plt.title('The distribution of language in the given data')
fig49.savefig('./result_image/49_Distribution_of_language.jpg')

fig50 = plt.figure(figsize=(16, 10))
make_count_plot(df=train, x='language', title='The number of pet with different languages and Adoption Speed')
fig50.savefig('./result_image/50_impact_of_pet_language.jpg')

# # sentiment Score
fig51 = plt.figure()
sns.violinplot(x="AdoptionSpeed", y="score", hue="Type", data=train, palette=['red', 'blue'], split=True)
plt.title('The impact of sentiment score for AdoptionSpeed')
fig51.savefig('./result_image/51_impact_of_sentiment.jpg')
