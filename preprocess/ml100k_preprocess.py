import itertools
import numpy as np
import pandas as pd

# subset can be ["movie_user_rating", "movie_title_rating", "movie_rating", "user_rating", "rating"]
fold = 1
# subset = "movie_title_poster_user_rating"
subset = "movie_title_user_rating"

in_files = {
    "user-train": "../movielens/u.user",
    "movie-train": "../movielens/u.item",
    "rating-train": "../movielens/u{:}.base".format(fold),
    "rating-test": "../movielens/u{:}.test".format(fold),
    "glove-search": "../Glove/glove.txt"
}

out_files = {
    "scale": "../ml100k-processed/u{:}-{:}-scale.npy".format(fold, subset),
    "train": "../ml100k-processed/u{:}-{:}-train.npy".format(fold, subset),
    "test": "../ml100k-processed/u{:}-{:}-test.npy".format(fold, subset),
    "idencoders": "../ml100k-processed/u{:}-{:}-idencoder.npy".format(fold, subset),
    "titles": "../ml100k-processed/u{:}-{:}-titles.npy".format(fold, subset),
    "title_dict": "../ml100k-processed/u{:}-{:}-title-dict.npy".format(fold, subset),
    "title_weights": "../ml100k-processed/u{:}-{:}-weights.npy".format(fold, subset)
}

user_headers = ["userid", "age", "gender", "occupation", "zip"]
user_r = ["is of_" + h for h in ["age", "gender", "occupation", "zip"]]
movie_headers = ["movieid", "title", "release date", "video release date", "IMDb URL", "unknown", "action", "adventure",
                 "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film-noir", "horror",
                 "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"]
movie_r = ["is of_" + h for h in ["title", "release date", "genre", "poster"]]
rating_headers = ["userid", "movieid", "rating", "timestamp"]


def read_and_filter():
    userdf = pd.read_csv(in_files["user-train"], engine="c", names=user_headers, sep="|")
    moviedf = pd.read_csv(in_files["movie-train"], engine="c", names=movie_headers, sep="|", encoding="latin1")
    rating_train = pd.read_csv(in_files["rating-train"], engine="c", names=rating_headers, sep="\t")
    rating_test = pd.read_csv(in_files["rating-test"], engine="c", names=rating_headers, sep="\t")

    # Normalize user ages
    age_scale_params = {
        "mean": userdf.mean()["age"],
        "std": userdf.std()["age"]
    }
    userdf["age"] -= age_scale_params["mean"]
    userdf["age"] /= age_scale_params["std"]

    # Slice first 2 digits of zip codes
    userdf["zip"] = userdf["zip"].str.slice(0, 2)

    # Normalize movie release dates
    moviedf["release date"] = pd.to_datetime(moviedf["release date"]).astype("int64")
    date_scale_params = {
        "mean": moviedf.mean()["release date"],
        "std": moviedf.std()["release date"]
    }
    moviedf["release date"] -= date_scale_params["mean"]
    moviedf["release date"] /= date_scale_params["std"]

    # Remove year from movie titles(And clear wrong data)
    moviedf["title"] = moviedf["title"].str.replace(r" \([0-9]+\)$", "")
    moviedf["title"] = moviedf["title"].str.replace(r" \([0-9]+\)$", "")
    moviedf["title"] = moviedf["title"].str.replace(r"[^a-zA-Z0-9 ]", " ")
    moviedf["title"] = moviedf["title"].str.replace(r"  ", " ")
    moviedf["title"] = moviedf["title"].str.replace(r"   ", " ")
    moviedf["title"] = moviedf["title"].str.replace(r"    ", " ")
    moviedf["title"] = moviedf["title"].str.replace(r" $", "")
    moviedf["title"] = moviedf["title"].str.replace(r"  $", "")
    moviedf["title"] = moviedf["title"].str.replace(r"   $", "")

    moviedf["title"] = moviedf["title"].str.replace(r"tianshi", "tian shi")
    moviedf["title"] = moviedf["title"].str.replace(r"wansui", "wan sui")
    moviedf["title"] = moviedf["title"].str.replace(r"VictorVictoria", "Victor Victoria")
    moviedf["title"] = moviedf["title"].str.replace(r"Jungle2Jungle", "Jungle 2 Jungle")
    moviedf["title"] = moviedf["title"].str.replace(r"Dadetown", "Dade town")
    moviedf["title"] = moviedf["title"].str.replace(r"Babyfever", "Baby fever")
    moviedf["title"] = moviedf["title"].str.replace(r"Lashou", "La shou")
    moviedf["title"] = moviedf["title"].str.replace(r"waipo", "wai po")
    moviedf["title"] = moviedf["title"].str.replace(r"Tianguo", "Tian guo")
    moviedf["title"] = moviedf["title"].str.replace(r"niezi", "nie zi")
    moviedf["title"] = moviedf["title"].str.replace(r"Quantestorie", "Quante storie")
    moviedf["title"] = moviedf["title"].str.replace(r"Aiqing", "Ai qing")
    moviedf["title"] = moviedf["title"].str.replace(r"Huozhe", "Huo zhe")
    moviedf["title"] = moviedf["title"].str.replace(r"Duoluo", "Duo luo")
    moviedf["title"] = moviedf["title"].str.replace(r"shentan", "shen tan")


    scale_params = {
        "age": age_scale_params,
        "date": date_scale_params
    }
    np.save(out_files["scale"], np.array(scale_params))

    return userdf, moviedf, rating_train, rating_test, scale_params


def build_dict(userdf, moviedf, rating_train, rating_test):
    genders = set(userdf["gender"])
    gender2id = dict(zip(genders, range(len(genders))))

    occupations = set(userdf["occupation"])
    job2id = dict(zip(occupations, range(len(occupations))))

    zipcodes = set(userdf["zip"])
    zip2id = dict(zip(zipcodes, range(len(zipcodes))))

    # chars = set(itertools.chain.from_iterable(moviedf["title"].values))
    # chars.update(["<go>", "<eos>"])
    # char2id = dict(zip(chars, range(len(chars))))
    # print(char2id)

    title_list = []
    for line in moviedf["title"]:
        line_list = line.split(' ')
        for i in line_list:
            title_list.append(i)
    chars = set((title_list))
    title_word = [i for i in chars]
    # title_word.remove('')
    chars = set(title_word)
    chars.update(["<go>", "<eos>"])
    char2id = dict(zip(chars, range(len(chars))))
    print(char2id)



    relations = set("rate_{:}".format(rating) for rating in set(rating_train["rating"]))
    relations.update(user_r)
    relations.update(movie_r)
    rel2id = dict(zip(relations, range(len(relations))))

    idenc = {
        "gender2id": gender2id,
        "job2id": job2id,
        "zip2id": zip2id,
        "char2id": char2id,
        "rel2id": rel2id,
        "maxuserid": max(userdf["userid"]),
        "maxmovieid": max(moviedf["movieid"])
    }

    np.save(out_files["idencoders"], np.array(idenc))

    return gender2id, job2id, zip2id, char2id, rel2id


def encode(userdf, moviedf, rating_train, rating_test, gender2id, job2id, zip2id, char2id, rel2id):
    train_triplets = []
    test_triplets = []
    title_symlist = []
    title_idlist = []
    title_weights = []
    attr2enc = {
        "gender": gender2id,
        "occupation": job2id,
        "zip": zip2id
    }

    af = "is of_"
    # Encode user attributes
    if "user" in subset:
        for attribute in ["age", "gender", "occupation", "zip"]:
            userids = userdf["userid"]
            attrs = userdf[attribute]
            for e1, e2 in zip(userids, attrs):
                encoded_e2 = attr2enc[attribute][e2] if attribute in attr2enc else e2
                train_triplets.append((e1, rel2id[af + attribute], encoded_e2))

    if "movie" in subset:
        movieids = moviedf["movieid"]
        if "title" in subset:
            # Encode movie titles
            titles = moviedf["title"]
            for e1, e2 in zip(movieids, titles):
                e2_list = e2.split(' ')
                encoded_e2 = [char2id["<go>"]] + [char2id[c] for c in e2_list] + [char2id["<eos>"]]
                train_triplets.append((e1, rel2id[af + "title"], encoded_e2))
                title_symlist.append(encoded_e2)
                title_idlist.append(e1)

        # Encode movie release dates
        print(len(train_triplets[3]))
        release_date = moviedf["release date"]
        for e1, e2 in zip(movieids, release_date):
            train_triplets.append((e1, rel2id[af + "release date"], e2))

        # Encode movie genres
        genre = moviedf[["unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary",
                         "drama", "fantasy", "film-noir", "horror", "musical", "mystery", "romance", "sci-fi", "thriller",
                         "war", "western"]]
        for e1, e2 in zip(movieids, genre.values):
            train_triplets.append((e1, rel2id[af + "genre"], e2))

    if "poster" in subset:
        poster_dict = np.load(in_files["cached-posters"]).item()
        for e1, e2 in poster_dict.items():
            train_triplets.append((e1, rel2id[af + "poster"], e2))

    # Encode training ratings
    for e1, e2, r, _ in rating_train.values:
        encoded_r = rel2id["rate_{:}".format(r)]
        train_triplets.append((e1, encoded_r, e2))

    # Encode test ratings
    for e1, e2, r, _ in rating_test.values:
        encoded_r = rel2id["rate_{:}".format(r)]
        test_triplets.append((e1, encoded_r, e2))

    training_set = np.array(train_triplets, dtype=tuple)
    test_set = np.array(test_triplets, dtype=tuple)
    title_set = np.array(title_symlist, dtype=list)
    title_dict = dict(zip(title_idlist, title_symlist))

    np.random.shuffle(training_set)
    np.random.shuffle(test_set)



    # Get random weights
    random_weights = []
    line_weights = [0 for i in range(300)]
    count = 0
    for char,cid in char2id.items():
        print(char)
        with open('../Glove/glove.txt', 'r+', encoding='utf-8') as csvfile:
            for line in csvfile:
                line_list = line.split(' ')
                if char ==line_list[0]:
                    count += 1
                    for i in range(1,301):
                        line_weights[i-1] += float(line_list[i])
                    break
    for weights in line_weights:
        random_weights.append(weights/count)
    print('--------------------------------------------------')


    # Extract title weights

    for char,cid in char2id.items():
        i = 0
        print(char)
        with open('../Glove/glove.txt', 'r+', encoding='utf-8') as csvfile:
            for line in csvfile:
                line_list = line.split(' ')
                # print(line_list[0])
                if char == line_list[0]:
                    i += 1
                    title_weights.append(line_list[1:])
                    break
        if i == 0:
            title_weights.append(random_weights)
    print(title_weights)


    np.save(out_files["test"], test_set)
    np.save(out_files["train"], training_set)
    np.save(out_files["titles"], title_set)
    np.save(out_files["title_dict"], title_dict)
    np.save(out_files["title_weights"], title_weights)


if __name__ == "__main__":
    userdf, moviedf, rating_train, rating_test, scale_params = read_and_filter()
    gender2id, job2id, zip2id, char2id, rel2id = build_dict(userdf, moviedf, rating_train, rating_test)
    encode(userdf, moviedf, rating_train, rating_test, gender2id, job2id, zip2id, char2id, rel2id)
