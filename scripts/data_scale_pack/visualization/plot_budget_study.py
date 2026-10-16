#!/usr/bin/env python3
"""Render verified controlled-budget runs/studies on haiqin; never overwrite outputs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.data_scale_pack.visualization.budget_results import load_budget_run, load_budget_study, LIMITATIONS, POLICIES
from scripts.data_scale_pack.localization_networks.workflow.tasks import BUDGET_ARCHITECTURES

COLORS = ("#0072B2", "#A65300", "#007A5E", "#8E4585", "#B13624")
MARKERS = ("o", "s", "^", "D", "X")
LABELS = ("Small: 3 blocks", "Medium: 4 blocks", "Large: 6 blocks", "XLarge: 8 blocks", "Baseline: 4 original blocks")
RENDERER_PATHS = (
    "scripts/data_scale_pack/visualization/plot_budget_study.py", "scripts/data_scale_pack/visualization/budget_results.py",
    "scripts/data_scale_pack/visualization/results.py", "scripts/data_scale_pack/visualization/study_results.py",
    "scripts/data_scale_pack/localization_networks/common/scaling.py",
    "scripts/data_scale_pack/localization_networks/workflow/budget_study.py",
    "scripts/data_scale_pack/localization_networks/workflow/calibration.py",
    "scripts/data_scale_pack/localization_networks/workflow/config.py", "scripts/data_scale_pack/localization_networks/workflow/tasks.py",
    "scripts/data_scale_pack/localization_networks/workflow/worker.py", "scripts/run_identity.py", "scripts/packs.py",
)
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.grid": True,
         "grid.alpha": .18, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
         "legend.edgecolor": "#b0b0b0", "legend.framealpha": 1, "legend.fancybox": False}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":"), allow_nan=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def renderer_provenance(source_files, training_hash, allow_update=False):
    expected = {item["path"]: item["sha256"] for item in source_files}
    actual = [{"path": path, "sha256": hashlib.sha256((ROOT/path).read_bytes()).hexdigest()} for path in RENDERER_PATHS]
    changed = [item["path"] for item in actual if item["sha256"] != expected.get(item["path"])]
    if changed and not allow_update:
        raise ValueError("Renderer differs from training source: " + ", ".join(changed) + "; use --allow-renderer-update only for explicit study redraws")
    return {"training_source_sha256": training_hash, "renderer_sources": actual,
            "renderer_source_sha256": hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "training_renderer_sources": [{"path": p, "sha256": expected.get(p)} for p in RENDERER_PATHS],
            "matches_training_snapshot": not changed, "changed_paths": changed, "allow_renderer_update": allow_update,
            "retrained": False, "action": "aggregate recorded evidence and redraw; no training or prediction"}


def prepare_output(output, sources, dpi):
    if type(dpi) is not int or not 72 <= dpi <= 600:
        raise ValueError("Preview dpi must be an integer from 72 to 600")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Budget output must be a new directory")
    if any(Path(item["path"]).resolve().is_relative_to(output.resolve()) for item in sources):
        raise ValueError("Budget output cannot contain its input evidence")
    return output


def file_manifest(output):
    return [{"path": p.relative_to(output).as_posix(), "size": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
             "group": "figure" if p.suffix in {".pdf", ".png"} else "data"}
            for p in sorted(output.rglob("*")) if p.is_file()]


def render_budget_run(run_dir: Path, output_dir: Path, dpi=160, *, require_pass=True):
    data = load_budget_run(run_dir, require_pass=require_pass)
    absolute = [{**s, "path": str(data.run_dir/s["path"])} for s in data.sources]
    output = prepare_output(output_dir, absolute, dpi)
    training_manifest = json.loads((data.run_dir/"manifest.json").read_text())
    provenance = renderer_provenance(training_manifest["files"], data.report["source_sha256"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=False)
    (output/"source-report.json").write_bytes(data.report_bytes)
    figures = []
    def save(fig, stem):
        path = output/(stem+".pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="pdf", dpi=dpi, facecolor="white")
        figures.append(path.relative_to(output).as_posix())
        plt.close(fig)
    try:
        with plt.rc_context(STYLE):
            fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))
            fig.subplots_adjust(bottom=.26, top=.79, wspace=.28)
            for index, (model, row) in enumerate(zip(data.models, data.observations, strict=True)):
                steps = [record["optimizer_steps"] for record in model.history]
                axes[0].plot(steps, [r["val_rmse_km"] for r in model.history], color=COLORS[index], marker=MARKERS[index], markevery=max(1, len(steps)//8), markersize=4, label=LABELS[index])
                axes[1].scatter(row["parameter_count"], row["test_rmse_km"], color=COLORS[index], marker=MARKERS[index], s=45)
            axes[0].set(xlabel="Optimizer updates", ylabel="Validation RMSE (km)")
            axes[1].set(xlabel="Trainable parameters", ylabel="Test RMSE (km)")
            axes[1].ticklabel_format(axis="x", style="sci", scilimits=(3,3))
            fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5,.07), ncol=3, fontsize=9)
            budget = data.observations[0]["budget_plan"]
            fig.suptitle(f"{data.report['run_id']}\n{budget['mode']} arm | N={data.observations[0]['train_count']} | {budget['total_optimizer_updates']} updates | {budget['validation_checks']} validation checks", fontsize=11)
            fig.text(.5,.025,"Five fixed profiles; test uses the first minimum-validation-RMSE checkpoint. No policy is trained here.",ha="center",fontsize=8)
            save(fig, "controlled-budget-summary")
            for index, (model, row) in enumerate(zip(data.models, data.observations, strict=True)):
                prefix = f"models/{model.method}/{model.model}"
                fig, ax = plt.subplots(figsize=(10,4.8), layout="constrained")
                ax.scatter([p.time_s for p in model.predictions], [p.true_km for p in model.predictions], marker="x", s=14, color="#222222", label="Label")
                ax.scatter([p.time_s for p in model.predictions], [p.predicted_km for p in model.predictions], marker=MARKERS[index], s=14, alpha=.65, color=COLORS[index], label="Prediction")
                ax.set(xlabel="Window center time (s)", ylabel="Range (km)", title=f"{model.caption}\n{data.report['run_id']} | best update {row['best_optimizer_step']}")
                ax.legend(fontsize=8)
                save(fig, prefix+"/test_outputs/prediction")
                fig, axes = plt.subplots(1,2,figsize=(11.6,4.8),layout="constrained")
                for ax, split, label in zip(axes,("train","val"),("Online training", "Validation"),strict=True):
                    ax.plot([r["optimizer_steps"] for r in model.history], [r[split+"_rmse_km"] for r in model.history],color=COLORS[index])
                    ax.axvline(row["best_optimizer_step"],color="#777777",linestyle="--",linewidth=.8,label="Selected checkpoint")
                    ax.set(xlabel="Optimizer updates",ylabel="RMSE (km)",title=label)
                    ax.legend(fontsize=8)
                fig.suptitle(f"{model.caption} | {len(model.history)} validation checkpoints\nOnline training is not a frozen-model training score",fontsize=11)
                save(fig,prefix+"/training/history")
    finally:
        plt.close("all")
    write_csv(output/"metrics.csv",data.observations)
    manifest = {"schema_version":1,"status":"PASS","pack":PACK,"run_id":data.report["run_id"],
                "created_at":datetime.now(timezone.utc).isoformat(),"suite":"budget","purpose":data.report["purpose"],
                "configuration":data.config,"config_sha256":data.report["config_sha256"],"source_sha256":data.report["source_sha256"],
                "uv_lock_sha256":data.report["uv_lock_sha256"],"selection":data.report.get("selection"),"model_plan":data.report["model_plan"],
                "run_dir":str(data.run_dir),"render_output_dir":str(output.resolve()),"formats":["pdf"],"dpi":dpi,
                "inputs":data.sources,"captured_report":"source-report.json","figures":figures,
                "models":[{**m.architecture,**m.metrics} for m in data.models],"renderer_sources":provenance["renderer_sources"],"renderer_provenance":provenance,
                "dependencies":{name:importlib.metadata.version(name) for name in ("matplotlib","numpy")},
                "delivery_roots":{"figure":f"output/figure/{data.report['run_id']}","data":f"output/data/{data.report['run_id']}/reports/visualization"},
                "limitations":LIMITATIONS,"files":file_manifest(output)}
    write_json(output/"manifest.json",manifest)
    return manifest


PACK = "data_scale_pack"


def render_budget_study(study_path: Path, output_dir: Path, *, runs_root=None, dpi=160, allow_renderer_update=False):
    import statistics
    from scripts.data_scale_pack.localization_networks.workflow import calibration
    from scripts.data_scale_pack.visualization.study_results import digest
    data = load_budget_study(study_path, runs_root)
    output = prepare_output(output_dir, data.sources, dpi)
    provenance = renderer_provenance(data.study["source_files"], data.study["source_sha256"], allow_renderer_update)
    artifact = data.study["calibration_artifact"]
    projection = calibration.validate_projection(artifact.get("projection"))
    if digest(projection, compact=True) != artifact.get("projection_sha256"):
        raise ValueError("Calibration development projection hash differs")
    fitted, candidates = calibration.fit_policy(projection)
    if fitted["policy"] != artifact["policy"] or fitted["score"] != artifact.get("score") or fitted["assignments"] != artifact.get("assignments"):
        raise ValueError("Recomputed development calibration differs from frozen artifact")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output.mkdir(parents=True, exist_ok=False)
    plan = data.study["plan"]
    arms, fractions, seeds = ("epochs", "updates"), sorted(plan["fractions"]), plan["seeds"]
    fixed = {(r["arm"],r["train_fraction"],r["architecture"]):r for r in data.summaries}
    policies = {(r["arm"],r["train_fraction"],r["policy_id"]):r for r in data.policy_summaries}
    references = {(r["arm"],r["train_fraction"]):r for r in data.selection}
    paired = {(r["comparison"],r["series"],r["arm"],r["train_fraction"]):r for r in data.paired_summaries}
    fixed_handles = [Line2D([],[],color=c,marker=m,label=l) for c,m,l in zip(COLORS,MARKERS,LABELS,strict=True)]
    policy_colors, policy_markers = ("#111111", "#8E4585"), ("*", "D")
    policy_labels = ("Original rule (600, 0.5)", f"Frozen calibrated rule ({artifact['policy']['reference_samples']}, {artifact['policy']['exponent']:g})")
    policy_handles = [Line2D([],[],color=c,marker=m,label=l) for c,m,l in zip(policy_colors,policy_markers,policy_labels,strict=True)]
    figures, captions = [], {}
    note = f"{len(seeds)} seeds: raw observations + mean ± sample SD, not CI. Full-N arms share execution; policies are virtual views."
    def grid(rows=1, cols=2, width=6.4, height=3.5):
        fig, axes = plt.subplots(rows,cols,figsize=(max(11.6,width*cols),height*rows+1.8),squeeze=False)
        fig.subplots_adjust(left=.085,right=.97,top=.85,bottom=max(.20,1.8/fig.get_figheight()),wspace=.30,hspace=.40)
        return fig,axes
    def finish(fig,stem,title,caption,handles=None,*,context=None,figure_note=None,source_metadata=None):
        passed = sum(r["status"]=="PASS" for r in data.unique_rows)
        if context is None:
            context=f"{data.study['study_id']} | {data.status}: {passed}/{len(data.unique_rows)} unique training observations"
        figure_note=note if figure_note is None else figure_note
        fig.suptitle(f"{title}\n{context}",fontsize=12,y=.98)
        if handles:
            fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.065),ncol=min(3,len(handles)),fontsize=9)
        fig.text(.5,.022,figure_note,ha="center",fontsize=8)
        for ext in ("pdf","png"):
            path=output/f"{stem}.{ext}"
            fig.savefig(path,format=ext,dpi=dpi,facecolor="white")
            figures.append(path.name)
        captions[stem]={"title":title,"description":caption,"width_inches":fig.get_figwidth(),"height_inches":fig.get_figheight(),"uncertainty":figure_note}
        if source_metadata is not None:
            captions[stem].update(context=context,source=source_metadata)
        plt.close(fig)
    def line(ax, groups, raw_groups, metric, color, marker, linestyle="-"):
        xs,ys=[],[]
        for summary,raw in zip(groups,raw_groups,strict=True):
            if not summary or not summary.get("n_seeds"):
                xs.append(float("nan"));ys.append(float("nan"));continue
            x,y=summary["train_count"],summary[metric+"_mean"]
            xs.append(x);ys.append(y)
            ax.scatter([r["train_count"] for r in raw],[r[metric] for r in raw],s=28,facecolors="none",edgecolors=color,marker=marker,alpha=.55)
            ax.errorbar(x,y,yerr=summary[metric+"_sd"],color=color,marker=marker,markersize=5,capsize=3,linestyle="none")
        ax.plot(xs,ys,color=color,linestyle=linestyle,linewidth=1.2)
    try:
        with plt.rc_context(STYLE):
            fig,axes=grid(height=5.5,width=7.2)
            observed=[r["best_val_rmse_km_mean"] for r in data.summaries if r["n_seeds"]]
            low,high=(min(observed),max(observed)) if observed else (0.,1.)
            if low==high: high=low+max(abs(low)*.1,1e-6)
            cmap=matplotlib.colormaps["cividis"].with_extremes(bad="#dddddd")
            for arm,ax in zip(arms,axes[0],strict=True):
                matrix=[[fixed[(arm,f,a)]["best_val_rmse_km_mean"] if fixed[(arm,f,a)]["n_seeds"] else float("nan") for f in fractions] for a in BUDGET_ARCHITECTURES]
                im=ax.imshow(matrix,cmap=cmap,vmin=low,vmax=high,aspect="auto")
                for y,a in enumerate(BUDGET_ARCHITECTURES):
                    for x,f in enumerate(fractions):
                        s=fixed[(arm,f,a)]
                        raw=[r for r in data.rows if r["status"]=="PASS" and r["arm"]==arm and r["train_fraction"]==f and r["architecture"]==a]
                        if s["n_seeds"]:
                            sd=f"±{s['best_val_rmse_km_sd']:.3f}" if s["best_val_rmse_km_sd"] is not None else "SD n/a"
                            observations="\n".join(f"{r['seed']}: {r['best_val_rmse_km']:.3f}" for r in raw)
                            label=f"{s['best_val_rmse_km_mean']:.3f} {sd}\n{observations}\nn={s['n_seeds']}/{len(seeds)}"
                        else: label="NOT_RUN"
                        ax.text(x,y,label,ha="center",va="center",fontsize=7.4,color="white" if s["n_seeds"] and matrix[y][x]<low+(high-low)*.43 else "#111111")
                labels=[]
                for a,label in zip(BUDGET_ARCHITECTURES,LABELS,strict=True):
                    s=next((fixed[(arm,f,a)] for f in fractions if fixed[(arm,f,a)]["n_seeds"]),None)
                    labels.append(label+(f"\nP={int(s['parameter_count_mean']):,}" if s else ""))
                ax.set(title=arm+" budget",xticks=range(len(fractions)),xticklabels=[f"N={fixed[(arm,f,BUDGET_ARCHITECTURES[0])]['train_count'] or 'NOT_RUN'}\nf={f:g}" for f in fractions],yticks=range(5),yticklabels=labels,xlabel="Retained training windows / fraction")
                ax.tick_params(axis="y",labelsize=8)
                ax.grid(False)
            fig.subplots_adjust(left=.14,right=.87,wspace=.50)
            cax=fig.add_axes((.89,.26,.014,.48))
            fig.colorbar(im,cax=cax,label="Mean best validation RMSE (km)")
            finish(fig,"validation-budget-heatmap","Capacity and validation performance under both budgets","Categorical architectures include actual parameter counts. Each cell shows mean ± sample SD followed by the individual values labeled by seed; missing cells stay gray. Full-data columns in the two panels are the same training observations.")

            fig,axes=grid(rows=2,cols=2,height=3.6)
            extra=[Line2D([],[],color=COLORS[0],marker="o",label="Fresh-validation-selected reference"),Line2D([],[],color=COLORS[4],marker="X",label="Original baseline architecture")]
            for ri,arm in enumerate(arms):
                series=[]
                for pi,pid in enumerate(POLICIES):
                    series.append(([policies[(arm,f,pid)] for f in fractions],[[r for r in data.policy_rows if r["status"]=="PASS" and r["arm"]==arm and r["train_fraction"]==f and r["policy_id"]==pid] for f in fractions],policy_colors[pi],policy_markers[pi],("--","-")[pi]))
                for which,color,marker in (("selected",COLORS[0],"o"),("baseline",COLORS[4],"X")):
                    groups,raws=[],[]
                    for f in fractions:
                        ref=references[(arm,f)]
                        arch=ref.get("selected_architecture") if which=="selected" else "complex_CNN_baseline"
                        groups.append(fixed.get((arm,f,arch)))
                        raws.append([r for r in data.rows if r["status"]=="PASS" and r["arm"]==arm and r["train_fraction"]==f and r["architecture"]==arch])
                    series.append((groups,raws,color,marker,":"))
                for si,(groups,raws,color,marker,style) in enumerate(series):
                    for ax,metric in zip(axes[ri],("parameter_count","test_rmse_km"),strict=True):
                        line(ax,groups,raws,metric,color,marker,style)
                    for fi,s in enumerate(groups):
                        if s and s["n_seeds"] and s["profiles"] != ["baseline"]:
                            axes[ri,0].annotate("/".join(s["profiles"]),(s["train_count"],s["parameter_count_mean"]),xytext=(0,(12,-18,28,-31)[si]),textcoords="offset points",ha="left" if fi==0 else "right" if fi==len(fractions)-1 else "center",color=color,fontsize=7.5)
                axes[ri,0].set(title=arm+": capacity choice",xlabel="Actual training windows N",ylabel="Trainable parameters")
                axes[ri,0].ticklabel_format(axis="y",style="sci",scilimits=(3,3));axes[ri,0].margins(y=.25,x=.12)
                axes[ri,1].set(title=arm+": held-out performance",xlabel="Actual training windows N",ylabel="Test RMSE (km)")
            finish(fig,"policy-capacity-performance","Frozen policy choices and observed quality","Policies map to existing fixed-profile observations; coincident points are not extra replicates. Reference uses mean fresh-validation RMSE over four scalable profiles; baseline is excluded. Lines connect tested budgets and do not define a new continuous rule.",policy_handles+extra)

            fig,axes=grid(height=4.0,width=6.8)
            comparison_handles=[]
            for ai,arm in enumerate(arms):
                key="calibrated-minus-original"
                groups=[paired[(key,"policy-comparison",arm,f)] for f in fractions]
                raws=[[r for r in data.paired_rows if r["status"]=="PASS" and r["comparison"]==key and r["arm"]==arm and r["train_fraction"]==f] for f in fractions]
                line(axes[0,0],groups,raws,"test_rmse_km_difference",policy_colors[ai],policy_markers[ai])
                comparison_handles.append(Line2D([],[],color=policy_colors[ai],marker=policy_markers[ai],label=f"Policy Δ: {arm}"))
            for i,arch in enumerate(BUDGET_ARCHITECTURES):
                groups=[paired[("updates-minus-epochs",arch,"updates",f)] for f in fractions]
                raws=[[r for r in data.paired_rows if r["status"]=="PASS" and r["comparison"]=="updates-minus-epochs" and r["series"]==arch and r["train_fraction"]==f] for f in fractions]
                line(axes[0,1],groups,raws,"test_rmse_km_difference",COLORS[i],MARKERS[i])
                comparison_handles.append(Line2D([],[],color=COLORS[i],marker=MARKERS[i],label="Budget Δ: "+LABELS[i]))
            for ax,title in zip(axes[0],("Calibrated − original policy","Equal updates − equal epochs"),strict=True):
                ax.axhline(0,color="#777777",linestyle="--",linewidth=.8);ax.set(title=title,xlabel="Actual training windows N",ylabel="Paired test RMSE difference (km)")
            finish(fig,"paired-policy-budget-differences","Within-seed policy and training-budget comparisons","Raw signed within-seed differences and their mean ± SD. Zero from identical policy mapping or full-N shared execution is structural, not an independent replication of equality. Negative means the first named condition has lower error.",comparison_handles)

            fig,axes=grid(rows=2,cols=2,height=3.4)
            for col,arm in enumerate(arms):
                for pi,pid in enumerate(POLICIES):
                    groups=[paired.get(("policy-minus-fresh-validation-reference",pid,arm,f)) for f in fractions]
                    raws=[[r for r in data.paired_rows if r["status"]=="PASS" and r["comparison"]=="policy-minus-fresh-validation-reference" and r["series"]==pid and r["arm"]==arm and r["train_fraction"]==f] for f in fractions]
                    for row,metric in enumerate(("best_val_rmse_km_difference","test_rmse_km_difference")):
                        line(axes[row,col],groups,raws,metric,policy_colors[pi],policy_markers[pi])
                for row,label in enumerate(("Signed validation RMSE difference (km)","Signed test RMSE difference (km)")):
                    axes[row,col].axhline(0,color="#777777",linestyle="--",linewidth=.8);axes[row,col].set(title=arm,xlabel="Actual training windows N",ylabel=label)
            finish(fig,"validation-reference-differences","Policy differences to the fresh-validation-selected reference","Reference is selected once per N/arm by cross-seed mean validation RMSE. Mean validation differences are nonnegative regret; individual seed differences may be negative and are never clipped. Test differences are signed comparisons, not test-oracle regret or independent policy validation.",policy_handles)

            fig,axes=grid(rows=2,cols=len(fractions),width=3.8,height=3.3)
            for ri,arm in enumerate(arms):
                for col,f in enumerate(fractions):
                    ax=axes[ri,col]
                    for ai,arch in enumerate(BUDGET_ARCHITECTURES):
                        ids={r["observation_id"] for r in data.rows if r["status"]=="PASS" and r["arm"]==arm and r["train_fraction"]==f and r["architecture"]==arch}
                        records=[r for r in data.histories if r["observation_id"] in ids]
                        steps=sorted({r["optimizer_steps"] for r in records})
                        means,spreads=[],[]
                        for step in steps:
                            values=[r["val_rmse_km"] for r in records if r["optimizer_steps"]==step]
                            means.append(statistics.mean(values));spreads.append(statistics.stdev(values) if len(values)>1 else 0.)
                        for seed in seeds:
                            individual=sorted((r for r in records if r["seed"]==seed),key=lambda r:r["optimizer_steps"])
                            ax.plot([r["optimizer_steps"] for r in individual],[r["val_rmse_km"] for r in individual],color=COLORS[ai],alpha=.18,linewidth=.65)
                        if steps:
                            ax.plot(steps,means,color=COLORS[ai],marker=MARKERS[ai],markevery=max(1,len(steps)//7),markersize=3)
                            if len({r["seed"] for r in records})>1: ax.fill_between(steps,[m-s for m,s in zip(means,spreads)],[m+s for m,s in zip(means,spreads)],color=COLORS[ai],alpha=.09)
                    ax.set(title=f"{arm} | f={f:g}"+(" | shared" if f==1 else ""),xlabel="Optimizer updates",ylabel="Validation RMSE (km)")
            finish(fig,"validation-update-trajectories","Validation trajectories in optimizer-update units","Thin lines are seeds; means and sample SD bands are evaluated at the actual common validation steps within each panel. Axes may have different ranges. The full-N panels duplicate one shared trajectory; no interpolation between validation checkpoints is used for aggregation.",fixed_handles)

            fig,axes=grid(rows=2,cols=len(fractions),width=3.8,height=3.3)
            for ri,arm in enumerate(arms):
                for col,f in enumerate(fractions):
                    ax=axes[ri,col]
                    for ai,arch in enumerate(BUDGET_ARCHITECTURES):
                        s=fixed[(arm,f,arch)]
                        if not s["n_seeds"]:continue
                        raw=[r for r in data.rows if r["status"]=="PASS" and r["arm"]==arm and r["train_fraction"]==f and r["architecture"]==arch]
                        ax.scatter([r["total_train_wall_s"] for r in raw],[r["test_rmse_km"] for r in raw],color=COLORS[ai],marker=MARKERS[ai],alpha=.4,s=25)
                        ax.errorbar(s["total_train_wall_s_mean"],s["test_rmse_km_mean"],xerr=s["total_train_wall_s_sd"],yerr=s["test_rmse_km_sd"],color=COLORS[ai],marker=MARKERS[ai],linestyle="none",capsize=2)
                    ax.set(title=f"{arm} | f={f:g}"+(" | shared" if f==1 else ""),xlabel="Observed training wall time (s)",ylabel="Test RMSE (km)")
            finish(fig,"budget-cost-quality","Training budget, observed cost, and held-out quality","Each panel shows fixed-profile observations, mean and horizontal/vertical sample SD. Times describe the recorded environment and are not controlled throughput benchmarks. Shared full-N observations are displayed twice but counted once in physical compute totals.",fixed_handles)

            references_grid=artifact["grid"]["reference_samples"]
            exponents=artifact["grid"]["exponents"]
            score={(c["policy"]["reference_samples"],c["policy"]["exponent"]):c["score"] for c in candidates}
            fig,axes=grid(rows=1,cols=1,width=12,height=5.8)
            ax=axes[0,0]
            matrix=[[score[(reference,exponent)] for reference in references_grid] for exponent in exponents]
            im=ax.imshow(matrix,origin="lower",aspect="auto",cmap="cividis_r")
            tick_indices=list(range(0,len(references_grid),max(1,len(references_grid)//10)))
            ax.set(xticks=tick_indices,xticklabels=[references_grid[i] for i in tick_indices],yticks=range(len(exponents)),yticklabels=[f"{e:g}" for e in exponents],xlabel="Reference sample count",ylabel="Scaling exponent")
            ax.grid(False)
            for policy,color,marker,label in (({"reference_samples":600,"exponent":.5},"#ffffff","o","Original rule"),(artifact["policy"],"#D55E00","*","Frozen calibrated rule")):
                ax.scatter(references_grid.index(policy["reference_samples"]),exponents.index(policy["exponent"]),marker=marker,s=150,facecolors=color,edgecolors="#111111" if marker=="o" else color,linewidths=1.8,label=label)
            ax.legend(loc="upper right",fontsize=9)
            fig.subplots_adjust(right=.85)
            cax=fig.add_axes((.88,.27,.018,.5));fig.colorbar(im,cax=cax,label="Mean relative validation regret (development)")
            development_seeds=artifact["development_seeds"]
            development_source={"study_id":artifact["source"]["study_id"],"development_seeds":development_seeds,
                                "validation_observations":len(projection),"candidate_count":len(candidates),"evidence_role":"development_validation_only"}
            development_context=(f"{development_source['study_id']} | {len(development_seeds)} development seeds ({', '.join(map(str,development_seeds))})\n"
                                 f"{len(projection)} validation observations | {len(candidates)} candidates")
            finish(fig,"development-calibration-grid","Development-only capacity-policy calibration","All 338 predefined candidates are recomputed from the frozen 48-row validation-only projection and checked against the saved selected policy, score and assignments. Equal-score regions may yield identical discrete profile mappings; the coefficients and thresholds are not uniquely identified. Fresh-study test metrics do not enter this plot.",
                   context=development_context,figure_note="Development validation only; test metrics excluded; no SD shown.",source_metadata=development_source)
    finally:
        plt.close("all")
    tables={"unique-training.csv":data.unique_rows,"analysis-cells.csv":data.rows,"summary.csv":data.summaries,
            "policy-views.csv":data.policy_rows,"policy-summary.csv":data.policy_summaries,
            "paired-differences.csv":data.paired_rows,"paired-summary.csv":data.paired_summaries,
            "validation-selection.csv":data.selection,"histories.csv":data.histories,
            "missing.csv":[r for r in data.rows if r["status"]!="PASS"],"calibration-candidates.csv":candidates}
    for filename,rows in tables.items():write_csv(output/filename,rows)
    write_json(output/"figure-captions.json",captions)
    write_json(output/"study-results.json",{"schema_version":1,"pack":PACK,"study_id":data.study["study_id"],"status":data.status,"study":data.study,"tables":tables,"limitations":LIMITATIONS})
    complete=[r for r in data.unique_rows if r["status"]=="PASS"]
    manifest={"schema_version":1,"pack":PACK,"study_id":data.study["study_id"],"status":data.status,
              "created_at":datetime.now(timezone.utc).isoformat(),"source_sha256":data.study["source_sha256"],"plan_sha256":data.study["plan_sha256"],
              "calibration_sha256":plan["calibration"]["sha256"],"renderer_provenance":provenance,"renderer_sources":provenance["renderer_sources"],
              "formats":["pdf","png"],"preview_dpi":dpi,"figures":figures,"inputs":data.sources,
              "counts":{"planned_unique_training":len(data.unique_rows),"completed_unique_training":len(complete),"logical_analysis_cells":len(data.study["analysis_cells"]),"logical_model_observations":len(data.rows),"virtual_policy_views":len(data.policy_rows)},
              "unique_compute":{"optimizer_updates":sum(r["optimizer_steps"] for r in complete),"sample_presentations":sum(r["sample_presentations"] for r in complete),"train_wall_s":sum(r["total_train_wall_s"] for r in complete)},
              "aggregation":{"replicate":"fresh seed, never policy/arm aliases","spread":"sample SD ddof=1, not CI","policy":"frozen mapping to fixed-profile observations","full_data":"shared execution, not independent arm observations","reference":"cross-seed fresh-validation-selected profile; signed test differences","missing":"explicit NOT_RUN, no imputation"},
              "dependencies":{name:importlib.metadata.version(name) for name in ("matplotlib","numpy")},"limitations":LIMITATIONS,"files":file_manifest(output)}
    write_json(output/"manifest.json",manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run",type=Path)
    source.add_argument("--study",type=Path)
    parser.add_argument("--runs-root",type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--dpi",type=int,default=160)
    parser.add_argument("--allow-renderer-update",action="store_true")
    args=parser.parse_args(argv)
    if args.run and (args.runs_root or args.allow_renderer_update):parser.error("--runs-root and --allow-renderer-update apply only to --study")
    try:
        result=render_budget_run(args.run,args.output_dir,dpi=args.dpi) if args.run else render_budget_study(args.study,args.output_dir,runs_root=args.runs_root,dpi=args.dpi,allow_renderer_update=args.allow_renderer_update)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.exit(1,f"Budget visualization failed: {exc}\n")
    print(f"{result['status']}: {len(result['figures'])} figures in {args.output_dir}")
    return 0 if result["status"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
