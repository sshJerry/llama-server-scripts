curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Write a detailed 1000-word essay about the history of computing, from Babbage to modern AI."}],"max_tokens":2048,"stream":false}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps({'timings':d.get('timings'),'usage':d.get('usage')},indent=2))" \
  > ~/35b-increment1-short.json