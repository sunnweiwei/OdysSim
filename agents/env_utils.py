import os
import re
import requests
import json
from openai import OpenAI

BASE_DOMAIN = os.getenv("BASE_DOMAIN")
RUNTIME_SERVICE_URL = os.getenv("RUNTIME_SERVICE_URL", f"http://{BASE_DOMAIN}:8005")


def extract_fn_call(text):
    """
    Extract function calls from text. Returns:
    - List of function call dicts if valid format found
    - {'error': 'message'} if wrong format detected
    - None if no function call found

    Function name must be: <function=name>
    Parameters can be: <parameter=name>value</parameter> OR <name>value</name>
    Both parameter formats can be mixed.
    """
    if not text:
        return None
    text = re.split(r'<\[[^\]]+\]>', text)[-1].strip()

    # Only accept <function=name> format for function names
    matches = list(re.finditer(
        r'(?m)^[ \t]*<function=([^>]+)>\s*(.*?)\s*</function>',
        text,
        re.DOTALL
    ))

    if not matches:
        # Check for incomplete function call
        fn_start = re.search(r'(?m)^[ \t]*<function=([^>]+)>', text)
        if fn_start:
            fn_name = fn_start.group(1)

            # Check for wrong closing tag format: </function=...> instead of </function>
            wrong_close_function = re.search(r'</function=', text)
            if wrong_close_function:
                return {
                    'error': f"""**Tool Call Format Error**

You used `</function=` as a closing tag, but closing tags should NOT have `=` in them.

**Your format (WRONG):**
```
<function={fn_name}>
...
</function={fn_name}>
```

**Correct format:**
```
<function={fn_name}>
...
</function>
```

The closing tag should simply be `</function>` without any `=` or name."""
                }

            # First check for wrong format: <parameter>value</parameter> instead of <parameter=name>value</parameter>
            wrong_param_format = re.findall(r'<parameter>([^<]*)</parameter>', text)
            if wrong_param_format:
                return {
                    'error': f"""**Tool Call Format Error**

You used `<parameter>` without specifying the parameter name. The parameter name must be in the opening tag.

**Your format (WRONG):**
```
<function={fn_name}>
<parameter>{wrong_param_format[0][:50]}...</parameter>
</function>
```

**Correct format:**
```
<function={fn_name}>
<parameter=message>{wrong_param_format[0][:50]}...</parameter>
</function>
```

Please use `<parameter=PARAM_NAME>value</parameter>` format. The parameter name (e.g., `message`, `command`, `path`) must be specified in the opening tag like `<parameter=message>`."""
                }

            open_params = len(re.findall(r'<parameter=[^>]+>', text))
            close_params = len(re.findall(r'</parameter>', text))
            has_close_function = bool(re.search(r'</function>', text))

            if (open_params != close_params) or (not has_close_function):
                return {
                    'error': f"""**Tool Call Format Error**

It looks like you started a tool call but didn't close one or more tags (e.g., missing `</parameter>` and/or `</function>`).

**Your format (WRONG):**
```
<function={fn_name}>
<parameter=query>...
<parameter=topk>10
```

**Correct format:**
```
<function={fn_name}>
<parameter=query>...</parameter>
<parameter=topk>10</parameter>
</function>
```

Please make sure every `<parameter=...>` has a matching `</parameter>`, and every `<function=...>` has a closing `</function>`."""
                }
        return None

    # Check for wrong parameter format and incomplete parameters within matched function calls
    for m in matches:
        fn_body = m.group(2)
        fn_name = m.group(1)

        # First check for wrong format: <parameter>value</parameter> instead of <parameter=name>value</parameter>
        wrong_param_format = re.findall(r'<parameter>([^<]*)</parameter>', fn_body)
        if wrong_param_format:
            preview = wrong_param_format[0][:50].replace('\n', ' ')
            return {
                'error': f"""**Tool Call Format Error**

You used `<parameter>` without specifying the parameter name. The parameter name must be in the opening tag.

**Your format (WRONG):**
```
<function={fn_name}>
<parameter>{preview}...</parameter>
</function>
```

**Correct format:**
```
<function={fn_name}>
<parameter=message>{preview}...</parameter>
</function>
```

Please use `<parameter=PARAM_NAME>value</parameter>` format. The parameter name (e.g., `message`, `command`, `path`) must be specified in the opening tag like `<parameter=message>`."""
            }

        # Check for incomplete parameters
        open_params = len(re.findall(r'<parameter=[^>]+>', fn_body))
        close_params = len(re.findall(r'</parameter>', fn_body))
        if open_params != close_params:
            return {
                'error': f"""**Tool Call Format Error**

It looks like you started a tool call but didn't close one or more parameter tags (e.g., missing `</parameter>`).

**Your format (WRONG):**
```
<function={fn_name}>
<parameter=query>...
```

**Correct format:**
```
<function={fn_name}>
<parameter=query>...</parameter>
</function>
```

Please make sure every `<parameter=...>` has a matching `</parameter>`."""
            }

    # Extract parameters - support both <parameter=name>value</parameter> and <name>value</name>
    groups = [[matches[0]]]
    for m in matches[1:]:
        prev = groups[-1][-1]
        line_gap = text.count('\n', prev.end(), m.start())
        groups[-1].append(m) if line_gap < 4 else groups.append([m])
    last = groups[-1]

    results = []
    for m in last:
        fn_body = m.group(2)
        fn_name = m.group(1)

        # Extract standard format parameters: <parameter=name>value</parameter>
        standard_params = dict(re.findall(
            r'<parameter=([^>]+)>(.*?)</parameter>',
            fn_body,
            re.DOTALL
        ))

        # Extract XML-style parameters: <name>value</name> (but exclude 'parameter' and 'function' tags)
        xml_params = re.findall(r'<([a-z_][a-z0-9_]*)>(.*?)</\1>', fn_body, re.DOTALL | re.IGNORECASE)
        xml_params_dict = {}
        for param_name, param_value in xml_params:
            # Skip 'parameter' and 'function' tags (these are structural, not parameters)
            if param_name.lower() not in ['parameter', 'function']:
                xml_params_dict[param_name] = param_value.strip()

        # Merge: standard params take precedence, then XML params
        merged_params = {**xml_params_dict, **standard_params}

        results.append({
            'name': fn_name,
            'arguments': merged_params
        })

    return results


def fn_call_to_text(fn_call) -> str:
    """
    Convert a parsed function call (or list of calls) back to text format.

    Args:
        fn_call: Dict with 'name' and 'arguments' keys, or list of such dicts

    Returns:
        String in format:
          - single: <function=name>\n<parameter=key>value</parameter>\n</function>
          - list: multiple such blocks concatenated with two newlines
    """
    # If we got a list of function calls, convert each and join
    if isinstance(fn_call, list):
        parts = [fn_call_to_text(item) for item in fn_call]
        return "\n\n".join(parts)

    if not isinstance(fn_call, dict) or 'name' not in fn_call:
        raise ValueError("fn_call must be a dict with 'name' key or a list of such dicts")

    fn_name = fn_call['name']
    arguments = fn_call.get('arguments', {}) or {}

    # Build the function call text
    lines = [f"<function={fn_name}>"]

    # Add parameters
    for key, value in arguments.items():
        # Convert value to string, handle None
        if value is None:
            value_str = ""
        else:
            value_str = str(value)
        lines.append(f"<parameter={key}>{value_str}</parameter>")

    lines.append("</function>")

    return "\n".join(lines)


class RuntimeServiceError(Exception):
    """Exception raised when runtime service is unavailable after retries."""
    pass


class BaseEnv:
    # Retry configuration
    MAX_RETRY_TIME = 30  # Total time to retry in seconds
    RETRY_DELAY = 2  # Delay between retries in seconds

    def __init__(self, **params):
        self.params = params
        self.runtime_id = None
        self.meta_info = None
        self.env_type = "tau"  # Default, subclasses should override

    async def _request_with_retry(self, method, url, **kwargs):
        """Make HTTP request with retry logic (30s total timeout)."""
        import asyncio
        import time
        import aiohttp
        start_time = time.time()
        last_error = None
        attempt = 0
        timeout_val = kwargs.pop('timeout', 30)
        json_data = kwargs.pop('json', None)

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < self.MAX_RETRY_TIME:
                attempt += 1
                try:
                    async with session.request(
                        method, url, json=json_data,
                        timeout=aiohttp.ClientTimeout(total=timeout_val), **kwargs
                    ) as response:
                        response.raise_for_status()
                        return await response.json()
                except aiohttp.ClientConnectionError as e:
                    last_error = e
                    elapsed = time.time() - start_time
                    if elapsed + self.RETRY_DELAY < self.MAX_RETRY_TIME:
                        await asyncio.sleep(self.RETRY_DELAY)
                    else:
                        break
                except aiohttp.ClientResponseError:
                    # Non-connection errors (like 4xx, 5xx) - don't retry
                    raise

        raise RuntimeServiceError(
            f"Runtime service unavailable after {attempt} attempts over {int(time.time() - start_time)}s. "
            f"Last error: {last_error}"
        )

    async def ping(self):
        """Check if environment exists and is responsive."""
        if not self.runtime_id:
            return {'exists': False, 'has_ping': False, 'ping_result': None, 'meta_info': None,
                    'message': 'No runtime_id'}
        try:
            result = await self._request_with_retry(
                'POST', f"{RUNTIME_SERVICE_URL}/ping",
                json={"runtime_id": self.runtime_id}, timeout=30
            )
            if result.get('meta_info'):
                result['meta_info'] = json.loads(result['meta_info'])
            return result
        except RuntimeServiceError:
            return {'exists': False, 'has_ping': False, 'ping_result': None, 'meta_info': None,
                    'message': 'Runtime service unavailable'}
        except Exception as e:
            return {'exists': False, 'has_ping': False, 'ping_result': None, 'meta_info': None,
                    'message': f'Ping error: {str(e)}'}

    async def create(self, env_type=None):
        """Create new runtime environment with retry."""
        if env_type is None:
            env_type = self.env_type
        # Filter out None values from params to avoid JSON serialization issues
        params = {k: v for k, v in self.params.items() if v is not None}
        result = await self._request_with_retry(
            'POST', f"{RUNTIME_SERVICE_URL}/create",
            json={"env_type": env_type, "params": json.dumps(params)},
            timeout=300
        )
        self.runtime_id = result['runtime_id']
        self.meta_info = json.loads(result['meta_info'])
        return self.meta_info

    async def _execute_step(self, fn_call: dict):
        """Execute single step with retry."""
        result = await self._request_with_retry(
            'POST', f"{RUNTIME_SERVICE_URL}/step",
            json={"runtime_id": self.runtime_id, "params": json.dumps(fn_call)},
            timeout=600
        )
        return result['result']

    async def get_reward(self, **kwargs):
        """Get reward from environment with retry."""
        result = await self._request_with_retry(
            'POST', f"{RUNTIME_SERVICE_URL}/reward",
            json={"runtime_id": self.runtime_id, "params": json.dumps(kwargs)},
            timeout=30
        )
        return result['reward']

    async def restore(self, conversation):
        """Restore environment by replaying actions from conversation."""
        # Parse actions from conversation
        actions = []
        for msg in conversation:
            if msg['role'] == 'assistant':
                fn_calls = extract_fn_call(msg['content'])
                if fn_calls:
                    actions.extend(fn_calls)

        # Recreate environment and replay
        await self.create()
        for fn_call in actions:
            try:
                await self._execute_step(fn_call)
            except Exception:
                pass  # Continue replaying even if some steps fail

    async def step(self, fn_call: dict, conversation=None):
        """Execute step with automatic recovery."""
        if not self.runtime_id:
            await self.create()

        # Ping and restore if needed
        if not (await self.ping())['exists']:
            if conversation:
                await self.restore(conversation)
            else:
                await self.create()

        # Execute step with retry
        try:
            return await self._execute_step(fn_call)
        except Exception:
            if conversation:
                await self.restore(conversation)
            else:
                await self.create()
            return await self._execute_step(fn_call)

    async def initialize(self, existing_runtime_id=None, conversation=None):
        """Initialize environment, validating existing_runtime_id or restoring from conversation.
        Only creates new runtime if existing one is invalid/missing.
        """
        # Try to use existing runtime_id if provided
        if existing_runtime_id:
            self.runtime_id = existing_runtime_id
            ping_result = await self.ping()
            if ping_result['exists']:
                # Existing runtime is valid - reuse it
                self.meta_info = ping_result.get('meta_info')
                return  # Don't create new runtime

        # Runtime doesn't exist or wasn't provided - need to create/restore
        # Check if we can restore from conversation (rare case - runtime went down)
        if conversation:
            has_actions = any(
                msg['role'] == 'assistant' and extract_fn_call(msg['content'])
                for msg in conversation
            )
            if has_actions:
                await self.restore(conversation)
                return  # restore() calls create() internally

        # No existing runtime and no actions to restore - create fresh
        await self.create()
