import requests
import json
import os
import time
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent / "output" / "statuspage_raw.json"

STATUSPAGE_URLS = [
    "https://www.githubstatus.com",
    "https://status.openai.com",
    "https://status.atlassian.com",
    "https://status.digitalocean.com",
    "https://status.datadoghq.com",
    "https://status.twilio.com",
    "https://status.dropbox.com",
    "https://status.newrelic.com",
    "https://status.shopify.com",
    "https://status.circleci.com",
    "https://status.sentry.io",
    "https://status.auth0.com",
    "https://status.squarespace.com",
    "https://www.redditstatus.com",
    "https://status.vercel.com",
    "https://www.netlifystatus.com",
    "https://status.pagerduty.com",
    "https://status.fastly.com",
    "https://discordstatus.com",
    "https://status.box.com",
]

def get_slug(url):
    return url.replace('https://', '').replace('.com', '').replace('www.', '').replace('status.', '').replace('.io', '')

def main():
    results = []
    for url in STATUSPAGE_URLS:
        slug = get_slug(url)
        print(f"Scraping company {slug}...")
        try:
            r = requests.get(f"{url}/api/v2/incidents.json", timeout=10)
            r.raise_for_status()
            data = r.json()
            incidents = data.get('incidents', [])
            count = 0
            for inc in incidents:
                if inc.get('status') != 'resolved':
                    continue
                updates = inc.get('incident_updates', [])
                if len(updates) < 2:
                    continue
                
                valid_update = False
                for u in updates:
                    if u.get('body') and len(u.get('body', '')) > 50:
                        valid_update = True
                        break
                if not valid_update:
                    continue
                    
                name = inc.get('name', '')
                if 'maintenance' in name.lower() or 'scheduled' in name.lower():
                    continue
                
                incident_id = f"{slug}-{inc.get('id')}"
                
                first_update = updates[-1] # Usually oldest is last in list
                resolved_update = updates[0] # Newest is first
                identified_update = None
                for u in updates:
                    if u.get('status') == 'identified':
                        identified_update = u
                        break
                
                raw_log = name + "\n" + (first_update.get('body') or '')
                root_cause = (identified_update.get('body') if identified_update else resolved_update.get('body')) or ''
                remediation = resolved_update.get('body') or ''
                source_url = f"{url}/incidents/{inc.get('id')}"
                
                if raw_log and root_cause and remediation:
                    results.append({
                        "incident_id": incident_id,
                        "raw_log_excerpt": raw_log.strip(),
                        "root_cause": root_cause.strip(),
                        "remediation_steps": remediation.strip(),
                        "source_url": source_url
                    })
                    count += 1
            print(f"  Got {count} incidents")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(1)
        
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as out:
        json.dump(results, out, indent=2)
    print(f"Saved {len(results)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
