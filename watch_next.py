import spacy

# Description of the last watched movie
last_movie = "Planet Hulk : Will he save their world or destroy it?" \
             " When the Hulk becomes too dangerous for the Earth," \
             " the Illuminati trick Hulk into a shuttle and launch him into space " \
             "to a planet where the Hulk can live in peace. Unfortunately, Hulk lands " \
             "on the planet Sakaar where he is sold into slavery and trained as a gladiator."

# Language model
nlp = spacy.load("en_core_web_md")

# Load the list of movies from the file to be used in the next_movie function
with open("movies.txt", "r") as file:
    movies = file.readlines()


def next_movie(last_description, list_movies):
    """
    :param last_description: Description of the last watched movie
    :param list_movies: Descriptions of suggested movies in the form of a list.
    :return: The movie most closely matched to the last watched one from the list of suggested movies.
    """
    similarity_dict = {}
    model_movies = nlp(last_description)
    for index, movie in enumerate(list_movies):
        similarity_dict[index] = nlp(movie).similarity(model_movies)

    return list_movies[max(similarity_dict, key=similarity_dict.get)]


if __name__ == "__main__":
    print(next_movie(last_movie, movies))
