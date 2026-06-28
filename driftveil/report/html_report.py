import html
from typing import Any

def generate_html_report(report: Any, path: str):
    """Generate a premium, responsive HTML report representing drift checks."""
    
    results = report.results
    total_checks = len(results)
    passed_checks = sum(1 for r in results if r.passed)
    failed_checks = sum(1 for r in results if not r.passed and r.severity == "error")
    warning_checks = sum(1 for r in results if not r.passed and r.severity == "warning")
    info_checks = sum(1 for r in results if not r.passed and r.severity == "info")
    
    overall_status = "PASSED" if report.passed else "FAILED"
    status_class = "badge-success" if report.passed else "badge-danger"
    
    # Generate table rows
    table_rows = []
    for r in results:
        # Target representation
        if r.target is None:
            target_str = "dataset"
        elif isinstance(r.target, tuple):
            target_str = f"{r.target[0]} &harr; {r.target[1]}"
        else:
            target_str = str(r.target)
            
        # Status icon
        if r.passed:
            status_icon = '<span class="icon icon-success">&check;</span>'
            row_class = "row-passed"
        else:
            if r.severity == "error":
                status_icon = '<span class="icon icon-danger">&cross;</span>'
                row_class = "row-failed"
            elif r.severity == "warning":
                status_icon = '<span class="icon icon-warning">&excl;</span>'
                row_class = "row-warning"
            else:
                status_icon = '<span class="icon icon-info">&iexcl;</span>'
                row_class = "row-info"
                
        severity_badge = f'<span class="badge-severity severity-{r.severity}">{r.severity}</span>'
        
        expected_safe = html.escape(str(r.expected))
        actual_safe = html.escape(str(r.actual))
        details_safe = html.escape(str(r.details))
        contract_safe = html.escape(str(r.contract))
        
        row_html = f"""
        <tr class="report-row {row_class}" data-passed="{str(r.passed).lower()}" data-severity="{r.severity}">
            <td>{status_icon}</td>
            <td><strong>{target_str}</strong></td>
            <td><code>{contract_safe}</code></td>
            <td>{details_safe}</td>
            <td>{expected_safe}</td>
            <td>{actual_safe}</td>
            <td>{severity_badge}</td>
        </tr>
        """
        table_rows.append(row_html)
        
    table_content = "\n".join(table_rows) if table_rows else '<tr><td colspan="7" class="empty-state">No checks executed</td></tr>'
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Driftveil - Data Drift Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151d30;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #1e293b;
            --success: #10b981;
            --success-bg: rgba(16, 185, 129, 0.1);
            --danger: #ef4444;
            --danger-bg: rgba(239, 68, 68, 0.1);
            --warning: #f59e0b;
            --warning-bg: rgba(245, 158, 11, 0.1);
            --info: #3b82f6;
            --info-bg: rgba(59, 130, 246, 0.1);
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            width: 100%;
            max-width: 1200px;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 40px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
        }}
        .logo-area h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }}
        .logo-area p {{
            color: var(--text-secondary);
            margin: 4px 0 0 0;
            font-size: 0.95rem;
        }}
        .badge {{
            padding: 8px 20px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .badge-success {{
            background-color: var(--success-bg);
            color: var(--success);
            border: 1px solid var(--success);
        }}
        .badge-danger {{
            background-color: var(--danger-bg);
            color: var(--danger);
            border: 1px solid var(--danger);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }}
        .stat-val {{
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 6px;
        }}
        .stat-val.passed {{ color: var(--success); }}
        .stat-val.failed {{ color: var(--danger); }}
        .stat-val.warnings {{ color: var(--warning); }}
        .stat-lbl {{
            color: var(--text-secondary);
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .control-panel {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .filters {{
            display: flex;
            gap: 10px;
        }}
        .filter-btn {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
        }}
        .filter-btn:hover {{
            border-color: var(--text-secondary);
            color: var(--text-primary);
        }}
        .filter-btn.active {{
            background-color: var(--text-primary);
            color: var(--bg-color);
            border-color: var(--text-primary);
            font-weight: 600;
        }}
        .report-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }}
        th {{
            background-color: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-color);
            padding: 18px 24px;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        td {{
            padding: 18px 24px;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.92rem;
            line-height: 1.5;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        .report-row {{
            transition: background-color 0.15s;
        }}
        .report-row:hover {{
            background-color: rgba(255, 255, 255, 0.015);
        }}
        .icon {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 26px;
            height: 26px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 0.9rem;
        }}
        .icon-success {{ background-color: var(--success-bg); color: var(--success); }}
        .icon-danger {{ background-color: var(--danger-bg); color: var(--danger); }}
        .icon-warning {{ background-color: var(--warning-bg); color: var(--warning); }}
        .icon-info {{ background-color: var(--info-bg); color: var(--info); }}
        
        .badge-severity {{
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }}
        .severity-error {{ background-color: var(--danger-bg); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.2); }}
        .severity-warning {{ background-color: var(--warning-bg); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.2); }}
        .severity-info {{ background-color: var(--info-bg); color: var(--info); border: 1px solid rgba(59, 130, 246, 0.2); }}
        
        .empty-state {{
            padding: 60px 24px;
            text-align: center;
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}
        code {{
            font-family: monospace;
            background-color: rgba(255, 255, 255, 0.05);
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 0.85rem;
            color: #f472b6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-area">
                <h1>Driftveil</h1>
                <p>Data Drift Detection Report</p>
            </div>
            <div>
                <span class="badge {status_class}">{overall_status}</span>
            </div>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val">{total_checks}</div>
                <div class="stat-lbl">Total Checks</div>
            </div>
            <div class="stat-card">
                <div class="stat-val passed">{passed_checks}</div>
                <div class="stat-lbl">Passed</div>
            </div>
            <div class="stat-card">
                <div class="stat-val failed">{failed_checks}</div>
                <div class="stat-lbl">Errors</div>
            </div>
            <div class="stat-card">
                <div class="stat-val warnings">{warning_checks}</div>
                <div class="stat-lbl">Warnings</div>
            </div>
        </div>
        
        <div class="control-panel">
            <div class="filters">
                <button class="filter-btn active" onclick="filterReport('all')">All</button>
                <button class="filter-btn" onclick="filterReport('passed')">Passed</button>
                <button class="filter-btn" onclick="filterReport('failed')">Failed (Errors)</button>
                <button class="filter-btn" onclick="filterReport('warning')">Warnings</button>
            </div>
        </div>
        
        <div class="report-card">
            <table>
                <thead>
                    <tr>
                        <th style="width: 5%"></th>
                        <th style="width: 20%">Target</th>
                        <th style="width: 15%">Contract</th>
                        <th style="width: 30%">Details</th>
                        <th style="width: 10%">Expected</th>
                        <th style="width: 10%">Actual</th>
                        <th style="width: 10%">Severity</th>
                    </tr>
                </thead>
                <tbody id="report-body">
                    {table_content}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        function filterReport(type) {{
            // Update filter button styling
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            const rows = document.querySelectorAll('.report-row');
            let hasVisible = false;
            
            rows.forEach(row => {{
                const passed = row.getAttribute('data-passed') === 'true';
                const severity = row.getAttribute('data-severity');
                
                if (type === 'all') {{
                    row.style.display = '';
                    hasVisible = true;
                }} else if (type === 'passed' && passed) {{
                    row.style.display = '';
                    hasVisible = true;
                }} else if (type === 'failed' && !passed && severity === 'error') {{
                    row.style.display = '';
                    hasVisible = true;
                }} else if (type === 'warning' && !passed && severity === 'warning') {{
                    row.style.display = '';
                    hasVisible = true;
                }} else {{
                    row.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
