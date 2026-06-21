curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a comprehensive 5000-word technical guide on implementing distributed systems, covering consensus algorithms, leader election, log replication, snapshotting, and failure recovery. Include code examples."}],"max_tokens":8192,"stream":false}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps({'timings':d.get('timings'),'usage':d.get('usage')},indent=2))" \
  > ~/35b-increment1-long.json