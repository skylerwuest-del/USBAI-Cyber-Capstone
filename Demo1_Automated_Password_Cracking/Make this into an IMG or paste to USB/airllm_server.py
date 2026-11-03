"""CPU AirLLM served locally to Hermes, with native tool-call conversion. No web UI."""
import argparse
import json
import os
from pathlib import Path
import signal
import select
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'airllm/site-packages'))
os.environ.update(HF_HOME=str(ROOT / 'airllm/cache'), HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                  HF_HUB_DISABLE_SYMLINKS_WARNING='1')
from airllm_experiment import load_streaming_model
from airllm_protocol import ProtocolError, completion, parse_output, prepare_messages, stream_chunks, validate


class CPUCleanup:
    """Weights still unload after each layer; expensive CPU GC runs once per pass."""
    def __init__(self, original, interval):
        self.original, self.interval, self.calls = original, max(1, interval), 0

    def __call__(self):
        self.calls += 1
        if self.calls >= self.interval:
            self.flush()

    def flush(self):
        self.calls = 0
        self.original()


class GenerationCancelled(Exception):
    pass


class CPUWeightCache:
    """Cache admitted CPU layers within a byte budget; other layers still stream."""
    def __init__(self, loader, dtype, budget):
        self.loader, self.dtype, self.budget = loader, dtype, max(0, budget)
        self.used, self.layers = 0, {}

    def __call__(self, name):
        if name in self.layers:
            return dict(self.layers[name])
        state = self.loader(name)
        cost = sum(value.numel() * (4 if value.is_floating_point() else value.element_size())
                   for value in state.values())
        if cost > self.budget - self.used:
            return state
        cached = {key: value.to(dtype=self.dtype) if value.is_floating_point() else value
                  for key, value in state.items()}
        self.layers[name] = cached
        self.used += cost
        return dict(cached)


class Engine:
    def __init__(self, path, context, max_tokens, cache_gb=8):
        import torch
        self.torch = torch
        self.model = load_streaming_model(path, context)
        # On CPU, module.to('meta') releases layer tensors through reference
        # counting. Collecting the entire Python heap 31+ times per generated
        # token adds seconds without being needed to unload those weights.
        # Scope this override to this separate CPU server process.
        import airllm.airllm_base as streaming
        from airllm.utils import clean_memory
        self.cleanup = CPUCleanup(clean_memory, len(self.model.layer_names))
        streaming.clean_memory = self.cleanup
        if not self.model.hf_quantizer and not self.model.compression:
            # Reserve room for the active layer, context KV cache, and the OS.
            meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            available = int(meminfo['MemAvailable'].split()[0]) * 1024
            cfg = getattr(self.model.config, 'text_config', self.model.config)
            heads = int(getattr(cfg, 'num_attention_heads', 1))
            kv_heads = int(getattr(cfg, 'num_key_value_heads', heads))
            head_dim = int(getattr(cfg, 'head_dim', None) or getattr(cfg, 'hidden_size', 1) // heads)
            kv_bytes = 2 * int(getattr(cfg, 'num_hidden_layers', 1)) * kv_heads * head_dim * context * 4
            largest = max(sum(p.numel() * 4 for p in layer.parameters()) for layer in self.model.layers)
            reserve = max(8 * 1024**3, kv_bytes + 2 * largest)
            budget = min(int(cache_gb * 1024**3), available // 3, max(0, available - reserve))
            self.weights = CPUWeightCache(self.model.load_layer_to_cpu, torch.float32, budget)
            self.model.load_layer_to_cpu = self.weights
            print('CPU weight cache budget: %.2f GB; larger uncached layers stream normally' % (budget / 1e9), flush=True)
        self.tokenizer = self.model.tokenizer
        if not self.tokenizer.chat_template:
            raise RuntimeError('Choose an instruction model with a tokenizer chat template')
        native_context = getattr(self.model.config, 'max_position_embeddings', context)
        self.context = min(context, native_context)
        self.max_tokens = max_tokens
        self.lock = threading.Lock()

    def generate(self, body, cancelled=None):
        if body.get('model', 'capstone') != 'capstone':
            raise ProtocolError('Use model name capstone')
        messages, tools = prepare_messages(body)
        if tools and 'tools' not in self.tokenizer.chat_template:
            messages[0]['content'] += '\n<tools>\n' + '\n'.join(json.dumps(t['function']) for t in tools) + '\n</tools>'
        text = self.tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors='pt', return_attention_mask=False)['input_ids']
        prompt_tokens = inputs.shape[1]
        requested = body.get('max_completion_tokens') or body.get('max_tokens') or self.max_tokens
        if not isinstance(requested, int) or requested < 1:
            raise ProtocolError('max_tokens must be a positive integer')
        budget = min(requested, self.max_tokens)
        if prompt_tokens + budget > self.context:
            raise ProtocolError('maximum context length exceeded: %d prompt tokens + %d output tokens > %d' %
                                (prompt_tokens, budget, self.context))
        temperature = float(body.get('temperature', 0.15))
        options = dict(max_new_tokens=budget, use_cache=True, return_dict_in_generate=True,
                       do_sample=temperature > 0,
                       pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                       attention_mask=self.torch.ones_like(inputs))
        if temperature > 0:
            options.update(temperature=temperature, top_p=0.8, top_k=20)
        started = time.monotonic()
        print('AirLLM request: %d prompt tokens; output limit %d' % (prompt_tokens, budget), flush=True)
        class Progress:
            def __init__(self):
                self.first, self.count = True, 0
            def put(self, value):
                if cancelled is not None and cancelled.is_set():
                    raise GenerationCancelled('Client disconnected; generation cancelled')
                if self.first:
                    self.first = False  # Transformers first supplies the input prompt.
                    return
                self.count += value.numel()
                if self.count == 1 or self.count % 32 == 0:
                    print('AirLLM generated %d tokens (%.1fs)' % (self.count, time.monotonic()-started), flush=True)
            def end(self):
                pass
        options['streamer'] = Progress()
        try:
            with self.torch.inference_mode():
                result = self.model.generate(inputs, **options)
        finally:
            self.cleanup.flush()
        generated = result.sequences[0, prompt_tokens:]
        output_tokens = len(generated)
        # Keep the model's tool delimiters; strip only end-of-message markers.
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        for marker in ('<|im_end|>', '<|endoftext|>', '<|eot_id|>'):
            text = text.replace(marker, '')
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        truncated = output_tokens >= budget and int(generated[-1]) not in eos_ids
        message, reason = parse_output(text, tools, body.get('tool_choice', 'auto'), truncated)
        response_format = body.get('response_format') or {}
        if reason == 'stop' and response_format.get('type') in ('json_object', 'json_schema'):
            content = message['content']
            content = __import__('re').sub(r'^```(?:json)?\s*|\s*```$', '', content.strip())
            try:
                value = json.loads(content)
                if response_format['type'] == 'json_schema':
                    validate(value, response_format['json_schema']['schema'])
            except ValueError as exc:
                raise ProtocolError('Model returned invalid structured JSON; retry the review') from exc
            message['content'] = json.dumps(value, ensure_ascii=False)
        print('AirLLM: %d prompt + %d generated tokens, %.1fs, %s' %
              (prompt_tokens, output_tokens, time.monotonic() - started, reason), flush=True)
        return completion(message, reason, prompt_tokens, output_tokens)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def send_json(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        engine = self.server.engine
        if self.path == '/health':
            self.send_json({'status': 'ok', 'backend': 'airllm', 'context_length': engine.context})
        elif self.path == '/v1/models':
            self.send_json({'object': 'list', 'data': [{'id': 'capstone', 'object': 'model',
                'created': 0, 'owned_by': 'local', 'context_length': engine.context,
                'max_model_len': engine.context, 'max_output_tokens': engine.max_tokens}]})
        elif self.path == '/props':
            self.send_json({'default_generation_settings': {'n_ctx': engine.context}})
        else:
            self.send_json({'error': {'message': 'Endpoint not found'}}, 404)

    def do_POST(self):
        if self.path != '/v1/chat/completions':
            self.send_json({'error': {'message': 'Endpoint not found'}}, 404)
            return
        if not self.server.engine.lock.acquire(blocking=False):
            self.close_connection = True  # Do not parse the unread body as another request.
            self.send_json({'error': {'message': 'AirLLM is processing another request'}}, 503)
            return
        done, cancelled = threading.Event(), threading.Event()
        watcher = None
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4 * 1024 * 1024:
                raise ProtocolError('Invalid request size')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ProtocolError('Request must be a JSON object')
            def watch_client():
                while not done.wait(0.25):
                    try:
                        readable, _, _ = select.select([self.connection], [], [], 0)
                        if readable and self.connection.recv(1, socket.MSG_PEEK) == b'':
                            cancelled.set()
                            return
                    except OSError:
                        cancelled.set()
                        return
            watcher = threading.Thread(target=watch_client, daemon=True)
            watcher.start()
            response = self.server.engine.generate(body, cancelled)
            if not body.get('stream'):
                self.send_json(response)
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.end_headers()
                for chunk in stream_chunks(response):
                    self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            pass
        except GenerationCancelled as exc:
            print(str(exc), flush=True)
            self.close_connection = True
        except (ValueError, KeyError, TypeError) as exc:
            self.close_connection = True
            self.send_json({'error': {'message': str(exc), 'type': 'invalid_request_error'}}, 400)
        except Exception as exc:
            print('AirLLM generation failed: ' + str(exc), flush=True)
            self.send_json({'error': {'message': str(exc), 'type': 'server_error'}}, 500)
        finally:
            done.set()
            if watcher is not None:
                watcher.join(timeout=1)
            self.server.engine.lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--context', type=int, default=16384)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--cache-gb', type=float, default=8)
    args = parser.parse_args()
    print('Loading AirLLM CPU model and layer files...', flush=True)
    engine = Engine(Path(args.model), args.context, args.max_tokens, args.cache_gb)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    server.engine = engine
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    print('AirLLM ready: localhost:%d, context %d' % (args.port, engine.context), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
