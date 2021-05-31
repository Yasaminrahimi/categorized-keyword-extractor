**Keyword Extractor**

In this model, our input is a spark dataframe, each row of which consists of a sentence and a category of this sentence.
And the output of this algorithm is the index words of each category.
In this code, the approximate chi algorithm is simulated.
The Chi algorithm scores the relevance of words and categories as follows using statistics to test the independence of two events.

*CHi2 Score* : ![Alt Text](https://www.thoughtco.com/thmb/qdhRNYCDTHsZk0b07td9AEe2nzk=/1071x346/filters:fill(auto,1)/chi-56b749505f9b5829f8380db4.gif)


The advantage of this method is that it eliminates general words between different categories and also
that it can be written for different languages and there is no need to define _stop words_.
Scalability, fast and distributed execution of this model are other advantages of this model
