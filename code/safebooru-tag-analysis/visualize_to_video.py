import matplotlib.animation as manimation
print(manimation.writers.list())

with open("safebooru-tag_trending-2010-2020.html","w") as fp1:
    s = '<!DOCTYPE html> \
<html> \
<body>\n \
{}\n\
<p>Credit:<a href=https://www.kaggle.com/alamson/safebooru>Safebooru Kaggle Dataset</a></p>\
</body> \
</html>\n'.format(myAnimation.to_html5_video())
    fp1.write(s)
print("done.")