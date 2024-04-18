# Preparation
# Import functions first
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Check the current operating version of TensorFlow
print(tf.__version__)

2.15.0

# Identify and analyze the dataset
# Load the dataset
data = pd.read_csv("/kaggle/input/spaceship-titanic/train.csv")
data.shape

(8693, 14)

# Print first five rows of the dataset
data.head(5)

PassengerId	HomePlanet	CryoSleep	Cabin	Destination	Age	VIP	RoomService	FoodCourt	ShoppingMall	Spa	VRDeck	Name	Transported
0	0001_01	Europa	False	B/0/P	TRAPPIST-1e	39.0	False	0.0	0.0	0.0	0.0	0.0	Maham Ofracculy	False
1	0002_01	Earth	False	F/0/S	TRAPPIST-1e	24.0	False	109.0	9.0	25.0	549.0	44.0	Juanna Vines	True
2	0003_01	Europa	False	A/0/S	TRAPPIST-1e	58.0	True	43.0	3576.0	0.0	6715.0	49.0	Altark Susent	False
3	0003_02	Europa	False	A/0/S	TRAPPIST-1e	33.0	False	0.0	1283.0	371.0	3329.0	193.0	Solam Susent	False
4	0004_01	Earth	False	F/1/S	TRAPPIST-1e	16.0	False	303.0	70.0	151.0	565.0	2.0	Willy Santantines	True

# Analyze the dataset features
data.describe()

Age	RoomService	FoodCourt	ShoppingMall	Spa	VRDeck
count	8514.000000	8512.000000	8510.000000	8485.000000	8510.000000	8505.000000
mean	28.827930	224.687617	458.077203	173.729169	311.138778	304.854791
std	14.489021	666.717663	1611.489240	604.696458	1136.705535	1145.717189
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
25%	19.000000	0.000000	0.000000	0.000000	0.000000	0.000000
50%	27.000000	0.000000	0.000000	0.000000	0.000000	0.000000
75%	38.000000	47.000000	76.000000	27.000000	59.000000	46.000000
max	79.000000	14327.000000	29813.000000	23492.000000	22408.000000	24133.000000

# Analyze the data type
data.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8693 entries, 0 to 8692
Data columns (total 14 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   PassengerId   8693 non-null   object 
 1   HomePlanet    8492 non-null   object 
 2   CryoSleep     8476 non-null   object 
 3   Cabin         8494 non-null   object 
 4   Destination   8511 non-null   object 
 5   Age           8514 non-null   float64
 6   VIP           8490 non-null   object 
 7   RoomService   8512 non-null   float64
 8   FoodCourt     8510 non-null   float64
 9   ShoppingMall  8485 non-null   float64
 10  Spa           8510 non-null   float64
 11  VRDeck        8505 non-null   float64
 12  Name          8493 non-null   object 
 13  Transported   8693 non-null   bool   
dtypes: bool(1), float64(6), object(7)
memory usage: 891.5+ KB

# Display the data visualization
plot_df = data.Transported.value_counts()
plot_df.plot(kind="bar")

<Axes: xlabel='Transported'>

# Format the graph structure
fig, ax = plt.subplots(5, 1, figsize=(10, 10))
plt.subplots_adjust(top=2)

# Create the graph feature
sns.histplot(data['Age'], color='b', bins=50, ax=ax[0])
sns.histplot(data['FoodCourt'], color='b', bins=50, ax=ax[1])
sns.histplot(data['ShoppingMall'], color='b', bins=50, ax=ax[2])
sns.histplot(data['Spa'], color='b', bins=50, ax=ax[3])
sns.histplot(data['VRDeck'], color='b', bins=50, ax=ax[4])

<Axes: xlabel='VRDeck', ylabel='Count'>

# Remove one label from the formatted dataset
data = data.drop(['PassengerId', 'Name'], axis=1)
data.head(5)

HomePlanet	CryoSleep	Cabin	Destination	Age	VIP	RoomService	FoodCourt	ShoppingMall	Spa	VRDeck	Transported
0	Europa	False	B/0/P	TRAPPIST-1e	39.0	False	0.0	0.0	0.0	0.0	0.0	False
1	Earth	False	F/0/S	TRAPPIST-1e	24.0	False	109.0	9.0	25.0	549.0	44.0	True
2	Europa	False	A/0/S	TRAPPIST-1e	58.0	True	43.0	3576.0	0.0	6715.0	49.0	False
3	Europa	False	A/0/S	TRAPPIST-1e	33.0	False	0.0	1283.0	371.0	3329.0	193.0	False
4	Earth	False	F/1/S	TRAPPIST-1e	16.0	False	303.0	70.0	151.0	565.0	2.0	True

# Advanced formatting
data.isnull().sum().sort_values(ascending=False)

CryoSleep       217
ShoppingMall    208
VIP             203
HomePlanet      201
Cabin           199
VRDeck          188
FoodCourt       183
Spa             183
Destination     182
RoomService     181
Age             179
Transported       0
dtype: int64

data[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = data[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
data.isnull().sum().sort_values(ascending=False)

HomePlanet      201
Cabin           199
Destination     182
RoomService     181
Age             179
CryoSleep         0
VIP               0
FoodCourt         0
ShoppingMall      0
Spa               0
VRDeck            0
Transported       0
dtype: int64

label = 'Transported'
data[label] = data[label].astype(int)

data['VIP'] = data['VIP'].astype(int)
data['CryoSleep'] = data['CryoSleep'].astype(int)

data[['Deck', 'Cabin_num', 'Side']] = data['Cabin'].str.split('/', expand=True)

try:
    data = data.drop('Cabin', axis=1)
except KeyError:
    print('Field does not exist')

# Print first five rows of the formatted dataset
data.head(5)

HomePlanet	CryoSleep	Destination	Age	VIP	RoomService	FoodCourt	ShoppingMall	Spa	VRDeck	Transported	Deck	Cabin_num	Side
0	Europa	0	TRAPPIST-1e	39.0	0	0.0	0.0	0.0	0.0	0.0	0	B	0	P
1	Earth	0	TRAPPIST-1e	24.0	0	109.0	9.0	25.0	549.0	44.0	1	F	0	S
2	Europa	0	TRAPPIST-1e	58.0	1	43.0	3576.0	0.0	6715.0	49.0	0	A	0	S
3	Europa	0	TRAPPIST-1e	33.0	0	0.0	1283.0	371.0	3329.0	193.0	0	A	0	S
4	Earth	0	TRAPPIST-1e	16.0	0	303.0	70.0	151.0	565.0	2.0	1	F	1	S

# Test and train the model
def split_dataset(dataset, test_ratio=0.20):
    test_indices = np.random.rand(len(dataset))<test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(data)
print("{} examples in training, {} examples in testing".format(
    len(train_ds_pd), len(valid_ds_pd)))

6958 examples in training, 1735 examples in testing

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label)

tfdf.keras.get_all_models()

[tensorflow_decision_forests.keras.RandomForestModel,
 tensorflow_decision_forests.keras.GradientBoostedTreesModel,
 tensorflow_decision_forests.keras.CartModel,
 tensorflow_decision_forests.keras.DistributedGradientBoostedTreesModel]

rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1")

Resolve hyper-parameter template "benchmark_rank1" to "benchmark_rank1@v1" -> {'winner_take_all': True, 'categorical_algorithm': 'RANDOM', 'split_axis': 'SPARSE_OBLIQUE', 'sparse_oblique_normalization': 'MIN_MAX', 'sparse_oblique_num_projections_exponent': 1.0}.
Use /tmp/tmpabvpxe3w as temporary training directory

rf = tfdf.keras.RandomForestModel()
rf.compile(metrics=['accuracy'])

Use /tmp/tmp9a1pj65c as temporary training directory

rf.fit(x=train_ds)

Reading training dataset...
Training dataset read in 0:00:08.894211. Found 6958 examples.
Training model...
  
[INFO 24-04-16 08:25:04.1755 UTC kernel.cc:1233] Loading model from path /tmp/tmp9a1pj65c/model/ with prefix fe0daa3d38cc4e4b

Model trained in 0:00:50.065102
Compiling model...

[INFO 24-04-16 08:25:05.4548 UTC decision_forest.cc:660] Model loaded with 300 root(s), 232766 node(s), and 13 input feature(s).
[INFO 24-04-16 08:25:05.4549 UTC abstract_model.cc:1344] Engine "RandomForestGeneric" built
[INFO 24-04-16 08:25:05.4549 UTC kernel.cc:1061] Use fast generic engine

Model compiled.

<tf_keras.src.callbacks.History at 0x7b85d32d8fd0>

tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)

import matplotlib.pyplot as plt

# Visualize the model testing and training summary
logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")
plt.show()

inspector = rf.make_inspector()
inspector.evaluation()

Evaluation(num_examples=6958, accuracy=0.7941937338315608, loss=0.5231154775835245, rmse=None, ndcg=None, aucs=None, auuc=None, qini=None)

evaluation = rf.evaluate(x=valid_ds, return_dict=True)

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")
          
2/2 [==============================] - 8s 90ms/step - loss: 0.0000e+00 - accuracy: 0.8006
loss: 0.0000
accuracy: 0.8006

print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
    print("\t", importance)
                                                          
Available variable importances:
	 NUM_NODES
	 SUM_SCORE
	 NUM_AS_ROOT
	 INV_MEAN_MIN_DEPTH

# Display index per feature
inspector.variable_importances()["NUM_AS_ROOT"]
[("CryoSleep" (1; #2), 125.0),
 ("RoomService" (1; #7), 66.0),
 ("Spa" (1; #10), 51.0),
 ("VRDeck" (1; #12), 32.0),
 ("ShoppingMall" (1; #8), 14.0),
 ("FoodCourt" (1; #5), 6.0),
 ("Deck" (4; #3), 4.0),
 ("Age" (1; #0), 1.0),
 ("HomePlanet" (4; #6), 1.0)]

# Generate a submission
# Load the test dataset
test_df = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
submission_id = test_df.PassengerId

# Replace NaN values with zero
test_df[['VIP', 'CryoSleep']] = test_df[['VIP', 'CryoSleep']].fillna(value=0)

# Format certain features
test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split("/", expand=True)
test_df = test_df.drop('Cabin', axis=1)

# Convert boolean to integers
test_df['VIP'] = test_df['VIP'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)

# Convert DataFrame to dataset
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df)

# Acquire predictions for test data
predictions = rf.predict(test_ds)
n_predictions = (predictions > 0.5).astype(bool)
output = pd.DataFrame({'PassengerId': submission_id, 'Transported': n_predictions.squeeze()})

# Show submission output
output.head()

5/5 [==============================] - 1s 84ms/step

PassengerId	Transported
0	0013_01	True
1	0018_01	False
2	0019_01	True
3	0021_01	True
4	0023_01	True

# Save as submission
sample_submission_df = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')
sample_submission_df['Transported'] = n_predictions
sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Successfully saved as submission file")

sample_submission_df.head()

Successfully saved as submission file

PassengerId	Transported
0	0013_01	True
1	0018_01	False
2	0019_01	True
3	0021_01	True
4	0023_01	True
