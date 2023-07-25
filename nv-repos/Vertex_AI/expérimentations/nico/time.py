# Import datetime class from datetime module
from datetime import datetime

#gs://nvallot_bucket/job_outputs/model/aiplatform-custom-training-2023-03-02-10:48:15.251

timestamp = datetime.now()
time = timestamp.isoformat()
text = time[:10] + '-' + time[11:]
delLastStr = text[:-3]
var3 = "aiplatform-custom-training-" + delLastStr
print(var3)