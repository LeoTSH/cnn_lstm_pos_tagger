## Data Visualization Project

# Summary:
Goal of this project is to create a visualization/infographics to relationships that influenced the survival rate of the passengers.
After analyzing the data, variables Passenger Class, Embarked Location and Sex affects survival rate with Passenger Class having to lowest effect.
Combination of the variables have a greater effect on survivability.

* Overwhelming likely to survive if you are a female
* More likely to survive if you are from Cherbourg
* More likely to survive if you are a First Class Passenger
* Combination of the mentioned variables seem to provide the most effect


# Design:
* Updated survival rates on all charts to percentages (Ration * 100)
* As the values of the survival rates were manually caluculated in the csv file and they considered a categorical type, I couldn't scale them on the Y-axis from 0 to 100 ate fixed intervals however, they are still arranged from smallest to largest
* Customized tooltip display
* Increased axes font size for easier reading
* For chart 3 (Location/Pclass), set legend to display full passenger class names and dropped sex as they will be separated by 2 bars in the chart
* Added short answers for readers to better comprehend the charts
* Some of the bars in the chart 3 are on the same level as they have the same survival rates for example in Queenstown, none of the male passengers from classes 1 and 2 survived
* Updated charts to display survival rate instead of numbers / percentages
* Updated all charts to display in Survival Rates on the Y-axis for easier reading
* Was unable to settle in a good way to place survival rate as it's kind of summary type and I couldn't find a way to place it on the Y-axis
* Remove default button, it was meant for users to toggle back to the default display page to compare between percentage and count values (No need for this section now)
* Color codes are default from dimple and I feel that they have enough contrast to differentiate and are not harsh on the eyes hence, I did not change them
* Updated chart axis titles to better reflect the variables they are representing
* Updated chart to display Surived 1 and 0 instead to show comparisons between survivors and victims
* Scatter-plots were dropped in favor of bar plots as they allow me to show instantaneously show the proportions of the data/variables, making the picture clearer
* Figures were set to display percentages for easier reading (Main page still displays numbers should readers need them for reference)
* Dropped Age from the charts as there are quite a number of empty entries for it
* From analysis, gender also played quite a part in survival and charts have been updated to reflect this 


# Feedback:
* Feedback 1: Circle radius in both the Default and Survivors Tab are the same, a location may have more survivors based on the amount of passengers, will be good to set radius to be a proportion of survivors/passengers

* Feedback 2: Visulizations show lack of direction on what story/information they are trying to convey

* Feedback 3: Try to color code plot points to show which category group they belong to
* Age is not suitable as there are blank entries


# Resources:
* https://bl.ocks.org/mbostock/3887118 (For scatter-plot chart)
* http://dimplejs.org/examples_index.html