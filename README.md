# nlp-json-reader
nlp json reader, copied from eden discord community

Matching Sequences
Running Solution
Warning: This will download ~1.5GB of data.

python3 -m similarity_matching --file=data/example.json




TODO 


Yeah, class abstraction would be fine. 

The model will always accept a certain type of data, called X. At the moment the program assumes all data comes from a Json file, via the training_data function in the data module. If you want that to use a database instead, how do we change it? 

Easy way is to have a base class with some definitions; 

class MyDataClass: 
  def __init__(self, topic_classifier: TopicClassifier) -> None:
    self.topic_classifier = topic_classifier

  @abstractmethod
  def row_data() -> Iterator[List[int, str, str, str]]:
    ...
  
  @abstractmethod
  def chunked_data() -> Iterator[pd.DataFrame]:
    ...

class FileImporter(MyDataClass):
  def __init__(self, file: pathlib.Path, topic_classifier: TopicClassifier) -> None:
    self.file = file
    super(FileImporter, self).__init__(topic_classifier)

  def row_data() -> Iterator[List[int, str, str, str]]:
    with open(self.file) as f:
      for line in f:
        yield _process_data(f, self.topic_classifier, ...)

class ElasticsearchImporter(MyDataClass):
  ...

class PostgresImporter(MyDataClass):
  ...


and you can kinda see the general pattern, being able to call any importer in __main__, likely via a factory function, and then use it regardless of what type of importer it is. So you treat your FileImporter identically to your Elasticsearch importer, etc.
