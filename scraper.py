import calendar
import datetime
import requests
import pickle as pkl


def get_posts(subreddit_name, start_time, end_time):

    timestamp = calendar.timegm(datetime.datetime.now().utctimetuple())
    # timestamp = calendar.timegm(datetime.datetime(2019, 12, 1).utctimetuple())
    # first = calendar.timegm(datetime.datetime(2018, 12, 1).utctimetuple())
    timestamp = end_time
    first = start_time
    posts = []
    print('begin downloading posts...')
    while timestamp > first:
        url = ("https://api.pushshift.io/reddit/search/submission/"
            "?subreddit={}&sort=desc&sort_type=created_utc&"
            "before={}&size=1000").format(subreddit_name,timestamp)
        r = requests.get(url).json()
        posts += r['data']
        print("Added posts from {} to {}".format(datetime.datetime.fromtimestamp(r['data'][-1]['created_utc']),
                                                datetime.datetime.fromtimestamp(r['data'][0]['created_utc'])))
        timestamp = r['data'][-1]['created_utc']

    pkl.dump(posts, open("./data/{}-posts.pkl".format(subreddit_name), "wb"))
    print('finished downloading ' + str(len(posts)) + ' posts...')
    return posts

def get_comments(subreddit_name, start_time, end_time):
    timestamp = calendar.timegm(datetime.datetime.now().utctimetuple())
    timestamp = end_time
    first = start_time
    comments = []
    while timestamp > first:
        url = ("https://api.pushshift.io/reddit/search/comment/"
            "?subreddit={}&sort=desc&sort_type=created_utc&"
            "before={}&size=1000").format(subreddit_name, timestamp)
        r = requests.get(url).json()
        comments += r['data']
        print("Added comments from {} to {}".format(datetime.datetime.fromtimestamp(r['data'][-1]['created_utc']),
                                                    datetime.datetime.fromtimestamp(r['data'][0]['created_utc'])))
        timestamp = r['data'][-1]['created_utc']

    pkl.dump(comments, open("./data/{}-comments.pkl".format(subreddit_name), "wb"))
    return comments
