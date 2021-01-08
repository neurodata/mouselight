# AWS Credentials Issues
 
:warning: **SECURITY DISCLAIMER** :warning:

Do **NOT** push any official AWS credentials to any repository. These posts are a good reference to get a sense of what pushing AWS credentials implies:

1. *I Published My AWS Secret Key to GitHub* by Danny Guo [https://www.dannyguo.com/blog/i-published-my-aws-secret-key-to-github/](https://www.dannyguo.com/blog/i-published-my-aws-secret-key-to-github/)
2. *Exposing your AWS access keys on Github can be extremely costly. A personal experience.* by Guru [https://medium.com/@nagguru/exposing-your-aws-access-keys-on-github-can-be-extremely-costly-a-personal-experience-960be7aad039](https://medium.com/@nagguru/exposing-your-aws-access-keys-on-github-can-be-extremely-costly-a-personal-experience-960be7aad039)
3. *Dev put AWS keys on Github. Then BAD THINGS happened* by Darren Pauli [https://www.theregister.com/2015/01/06/dev_blunder_shows_github_crawling_with_keyslurping_bots/](https://www.theregister.com/2015/01/06/dev_blunder_shows_github_crawling_with_keyslurping_bots/)


Brainlit can access data volumes stored in [AWS S3](https://aws.amazon.com/free/storage/s3/?trk=ps_a134p000006BgagAAC&trkCampaign=acq_paid_search_brand&sc_channel=ps&sc_campaign=acquisition_US&sc_publisher=google&sc_category=storage&sc_country=US&sc_geo=NAMER&sc_outcome=acq&sc_detail=aws%20s3&sc_content=S3_e&sc_segment=432339156183&sc_medium=ACQ-P|PS-GO|Brand|Desktop|SU|Storage|Product|US|EN|Text&s_kwcid=AL!4422!3!432339156183!e!!g!!aws%20s3&ef_id=CjwKCAjwkoz7BRBPEiwAeKw3q7yLVNTPLORSa7QUsB5aGT0wAKrnrlnkwNPex8vdqYMVBPqgjlZV2RoCIdgQAvD_BwE:G:s&s_kwcid=AL!4422!3!432339156183!e!!g!!aws%20s3) through the [CloudVolume](https://github.com/seung-lab/cloud-volume) package. As specified in the [docs](https://github.com/seung-lab/cloud-volume#credentials), AWS credentials have to be stored in a file called `aws-secret.json` inside the `~.cloudvolume/secrets/` folder.

Prerequisites to successfully troubleshoot errors related to AWS credentials:

- [ ] The data volume is hosted on S3 (i.e. the link looks like `s3://your-bucket-name/some-path/some-folder`).
- [ ] Familiarity with [IAM Roles](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) and [how to create them](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create.html).
- [ ] An `AWS_ACCESS_KEY_ID` and an `AWS_SECRET_ACCESS_KEY` with adequate permissions, provided by an AWS account administrator. Brainlit does not require the IAM user associated with the credentials to have access to the AWS console (i.e. it can be a service account).

Here is a collection of known issues, along with their troubleshoot guide:

## Missing `AWS_ACCESS_KEY_ID`

Error message:

```python
~/opt/miniconda3/envs/brainlit/lib/python3.8/site-packages/cloudvolume/connectionpools.py in _create_connection(self)
     99       return boto3.client(
    100         's3',
--> 101         aws_access_key_id=self.credentials['AWS_ACCESS_KEY_ID'],
    102         aws_secret_access_key=self.credentials['AWS_SECRET_ACCESS_KEY'],
    103         region_name='us-east-1',

KeyError: 'AWS_ACCESS_KEY_ID'
```

This error is thrown when the `credentials` object has an empty `AWS_ACCESS_KEY_ID` entry. This probably indicates that `aws-secret.json`  is not stored in the right folder and it cannot be found by CloudVolume. Make sure your credential file is named correctly and stored in `~.cloudvolume/secrets/`. If you are a Windows user, the output of this Python snippet is the expansion of `~` for your system:

```python
import os
HOME = os.path.expanduser('~')
print(HOME)
```

example output:

```bash
Python 3.8.3 (v3.8.3:6f8c8320e9)
>>> import os
>>> HOME = os.path.expanduser('~')
>>> print(HOME)
C:\Users\user
```

## Empty `AKID` (Access Key ID)

Error message:

```python
/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/botocore/client.py in _make_api_call(self, operation_name, api_params)
    654             error_code = parsed_response.get("Error", {}).get("Code")
    655             error_class = self.exceptions.from_code(error_code)
--> 656             raise error_class(parsed_response, operation_name)
    657         else:
    658             return parsed_response
ClientError: An error occurred (AuthorizationHeaderMalformed) when calling the GetObject operation: The authorization header is malformed; a non-empty Access Key (AKID) must be provided in the credential.
```

This error is thrown when your `aws-secret.json` file is stored and loaded correctly, and it looks like this:

```json
{
  "AWS_ACCESS_KEY_ID": "",
  "AWS_SECRET_ACCESS_KEY": ""
}
```

Even though the bucket itself may be public, [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) requires some non-empty AWS credentials to instantiante the S3 API client.

## Access denied

```python
/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/botocore/client.py in _make_api_call(self, operation_name, api_params)
    654             error_code = parsed_response.get("Error", {}).get("Code")
    655             error_class = self.exceptions.from_code(error_code)
--> 656             raise error_class(parsed_response, operation_name)
    657         else:
    658             return parsed_response
ClientError: An error occurred (AccessDenied) when calling the GetObject operation: Access Denied
```

This error is thrown when:

1. The AWS credentials are stored and loaded correctly but are not allowed to access the data volume. A check with an AWS account administrator is required.

2. There is a typo in your credentials. The content of `aws-secret.json` should look like this:

```json
{
  "AWS_ACCESS_KEY_ID": "$YOUR_AWS_ACCESS_KEY_ID",
  "AWS_SECRET_ACCESS_KEY": "$AWS_SECRET_ACCESS_KEY"
}
```

where the `$` are placeholder characters and should be replaced along with the rest of the string with the official AWS credentials.