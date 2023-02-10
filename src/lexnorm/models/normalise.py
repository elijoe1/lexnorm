# TODO remember always normalise 'rt' to 'retweet' if followed by @mention
# hard coded normalisation of 'rt' if followed by @mention following notebook 1.0 and 2015 annotation guideline 3.
# we can do this as 'rt' is a domain specific entity and normalisation is fairly deterministic (when in middle
# of tweet and not followed by @mention) and when normalised, always to 'retweet'
# if 0 < i < len(raw_tweet) - 1 and raw_tweet[i + 1][0] != "@":
