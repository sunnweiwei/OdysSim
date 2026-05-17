import re


def convert_tools_to_description(tools: list[dict]) -> str:
    ret = ''
    for i, tool in enumerate(tools):
        assert tool['type'] == 'function'
        fn = tool['function']
        if i > 0:
            ret += '\n'
        ret += f'---- BEGIN FUNCTION #{i + 1}: {fn["name"]} ----\n'
        ret += f'Description: {fn["description"]}\n'

        if 'parameters' in fn:
            ret += 'Parameters:\n'
            properties = fn['parameters'].get('properties', {})
            required_params = set(fn['parameters'].get('required', []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                # Indicate required/optional in parentheses with type
                is_required = param_name in required_params
                param_status = 'required' if is_required else 'optional'
                param_type = param_info.get('type', 'string')

                # Get parameter description
                desc = param_info.get('description', 'No description provided')

                # Handle enum values if present
                if 'enum' in param_info:
                    enum_values = ', '.join(f'`{v}`' for v in param_info['enum'])
                    desc += f'\nAllowed values: [{enum_values}]'

                ret += (
                    f'  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n'
                )
        else:
            ret += 'No parameters are required for this function.\n'

        ret += f'---- END FUNCTION #{i + 1} ----\n'
    return ret


def search_tool():
    search = {
        'type': 'function',
        'function': {
            "name": "search",
            "description": "Performs a web search: supply a string 'query'. The tool retrieves the results for the query, returning their url, title, and snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "The maximum number of search results to return (default 5)."
                    }
                },
                "required": ["query"]
            }
        }
    }
    extract = {
        'type': 'function',
        'function': {
            'name': 'extract',
            'description': (
                "Extract web page content from one or more specified URLs"
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'url': {
                        'type': 'string',
                        'description': 'The URL to extract content from.',
                    },
                    'query': {
                        'type': 'string',
                        'description': 'Optional: user intent for reranking extracted content chunks. When provided, chunks are reranked based on relevance to this query.',
                    },
                },
                'required': ['url'],
            },
        },
    }
    return [search, extract]

def get_mcp_tools(mcp_servers):
    mcp_tool_map = {}  # tool_name -> (server_id, server_config)
    mcp_tool_list = []
    if mcp_servers:
        print(f"[agent_loop] Loading MCP servers: {len(mcp_servers)} servers provided")
        try:
            from mcp_manager import mcp_manager
            import asyncio

            # Connect to MCP servers and get tools
            for server_config in mcp_servers:
                server_id = server_config.get("server_id")
                command = server_config.get("command", "")
                args = server_config.get("args", [])
                env = server_config.get("env")

                print(f"[agent_loop] Attempting to connect to MCP server: {server_id}")
                print(f"[agent_loop] Command: {command}, Args: {args}")

                try:
                    # Connect to server (this is async, so we need to run it)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    connected = loop.run_until_complete(
                        mcp_manager.connect_server(
                            server_id,
                            command,
                            args,
                            env
                        )
                    )

                    if connected:
                        # Get tools from this server
                        mcp_tools = mcp_manager.get_tools(server_id)
                        print(
                            f"[agent_loop] Successfully connected to MCP server {server_id}, found {len(mcp_tools)} tools")
                        for mcp_tool in mcp_tools:
                            # Convert to function schema
                            function_tool = mcp_manager.convert_mcp_tool_to_function_schema(mcp_tool)
                            mcp_tool_list.append(function_tool)
                            # Map tool name to server for execution
                            tool_name = mcp_tool.get("name", "")
                            if tool_name:
                                mcp_tool_map[tool_name] = (server_id, server_config)
                            print(f"[agent_loop] Added MCP tool: {tool_name}")
                        print(f"[agent_loop] Loaded {len(mcp_tools)} tools from MCP server {server_id}")
                    else:
                        print(f"[agent_loop] Failed to connect to MCP server {server_id}")
                    loop.close()
                except Exception as e:
                    import traceback
                    print(f"[agent_loop] Exception loading MCP server {server_id}: {e}")
                    traceback.print_exc()
                    print(f"[agent_loop] Server config: command={command}, args={args}, env={env}")
        except Exception as e:
            import traceback
            print(f"[agent_loop] Error loading MCP tools: {e}")
            traceback.print_exc()
    return mcp_tool_list, mcp_tool_map

def extract_fn_call(text):
    if not text:
        return None
    text = re.split(r'<\[[^\]]+\]>', text)[-1].strip()
    matches = list(re.finditer(r'(?m)^[ \t]*<function=([^>]+)>\s*(.*?)\s*</function>',
                               text, re.DOTALL))
    if not matches:
        return None
    groups = [[matches[0]]]
    for m in matches[1:]:
        prev = groups[-1][-1]
        line_gap = text.count('\n', prev.end(), m.start())
        groups[-1].append(m) if line_gap < 4 else groups.append([m])
    last = groups[-1]
    return [
        {
            'function': m.group(1),  # <-- each call uses its *own* captured fn name
            'arguments': dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>',
                                         m.group(2), re.DOTALL))
        }
        for m in last
    ]

TOOL_PROMPT = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Parameters must be wrapped with <parameter=key>value</parameter>
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""

PARALLEL_TOOL_PROMPT = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""

#
