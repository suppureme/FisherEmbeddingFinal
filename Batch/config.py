# -------------------------------------------------------------------------
#
# THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
# EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
# ----------------------------------------------------------------------------------
# The example companies, organizations, products, domain names,
# e-mail addresses, logos, people, places, and events depicted
# herein are fictitious. No association with any real company,
# organization, product, domain name, email address, logo, person,
# places, or events is intended or should be inferred.
# --------------------------------------------------------------------------

# Global constant variables (Azure Storage account/Batch details)

# import "config.py" in "python_quickstart_client.py "

_BATCH_ACCOUNT_NAME = 'acsysbatchservice'  # Your batch account name
_BATCH_ACCOUNT_KEY = 'l1Y0O2e6tt85ibZX0eTLmDsnOYAuO9zRSQeW9BJoxBjBfe8EolXPKxhOIPfp3CEYhDSrK+yOMtTuT8f0/zkLnA=='  # Your batch account key
_BATCH_ACCOUNT_URL = 'https://acsysbatchservice.centralindia.batch.azure.com'  # Your batch account URL
_STORAGE_ACCOUNT_NAME = 'acsysbatchstroageacc'  # Your storage account name
_STORAGE_ACCOUNT_KEY = 'A1I+gDNiz+bBbFbAh+vvRthOccMvAOQ8BkpJ8EVfp3x4yWnVe1fx4aW8BUSeovltcM4TBqb7YLEAOabjuLisdA=='  # Your storage account key
_POOL_ID = 'acsysbatchpool01'  # Your Pool ID
_POOL_NODE_COUNT = 1  # Pool node count    #single node for now
_POOL_VM_SIZE = 'STANDARD_D4_v3'  # VM Type/Size   #'STANDARD_A1_v2'
_JOB_ID = 'Test6'  # Job ID
_STANDARD_OUT_FILE_NAME = 'stdout.txt'  # Standard Output file
_DEDICATED_POOL_NODE_COUNT = 1
_LOW_PRIORITY_POOL_NODE_COUNT = 1