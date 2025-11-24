# Pipeline Dashboard Section - Insert this into your page == "Pipeline Board" section

elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    
    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        df = leads_df(s)
        weights = st.session_state.weights
        
        # Load ML model if exists
        try:
            lead_model = joblib.load('lead_conversion_model.pkl')
        except:
            lead_model = None
        
        # ==================== GOOGLE ADS-STYLE CARDS ====================
        st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 4px 6px rgba(0,0,0,0.3), 0 1px 3px rgba(0,0,0,0.2);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.4), 0 2px 4px rgba(0,0,0,0.3);
        }
        .metric-label {
            font-size: 13px;
            color: #93a0ad;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 32px;
            font-weight: 700;
            margin: 8px 0;
        }
        .metric-change {
            font-size: 13px;
            font-weight: 600;
            padding: 4px 8px;
            border-radius: 6px;
            display: inline-block;
        }
        .metric-change.positive {
            background: rgba(34, 197, 94, 0.15);
            color: #22c55e;
        }
        .metric-change.negative {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 12px;
        }
        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .stage-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin: 4px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Calculate metrics
        total_leads = len(df)
        qualified_leads = len(df[df['qualified'] == True])
        total_value = df['estimated_value'].sum()
        awarded_leads = len(df[df['status'] == LeadStatus.AWARDED])
        lost_leads = len(df[df['status'] == LeadStatus.LOST])
        
        # Conversion rate
        closed_leads = awarded_leads + lost_leads
        conversion_rate = (awarded_leads / closed_leads * 100) if closed_leads > 0 else 0
        
        # Stage counts
        stage_counts = df['status'].value_counts().to_dict()
        
        # Stage colors mapping
        stage_colors = {
            LeadStatus.NEW: "#2563eb",
            LeadStatus.CONTACTED: "#eab308",
            LeadStatus.INSPECTION_SCHEDULED: "#f97316",
            LeadStatus.INSPECTION_COMPLETED: "#14b8a6",
            LeadStatus.ESTIMATE_SUBMITTED: "#a855f7",
            LeadStatus.AWARDED: "#22c55e",
            LeadStatus.LOST: "#ef4444"
        }
        
        # Top row - Key metrics cards
        st.markdown("### 📊 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Leads</div>
                <div class="metric-value" style="color: #2563eb;">{total_leads}</div>
                <div class="metric-change positive">
                    ↑ {qualified_leads} Qualified
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Pipeline Value</div>
                <div class="metric-value" style="color: #22c55e;">${total_value:,.0f}</div>
                <div class="metric-change positive">
                    ↑ Active
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Conversion Rate</div>
                <div class="metric-value" style="color: #a855f7;">{conversion_rate:.1f}%</div>
                <div class="metric-change {'positive' if conversion_rate > 50 else 'negative'}">
                    {awarded_leads}/{closed_leads} Won
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            active_leads = total_leads - closed_leads
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Active Leads</div>
                <div class="metric-value" style="color: #f97316;">{active_leads}</div>
                <div class="metric-change {'positive' if active_leads > 0 else 'negative'}">
                    In Progress
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Stage breakdown cards
        st.markdown("### 📈 Pipeline Stages")
        
        stage_cols = st.columns(len(LeadStatus.ALL))
        for idx, stage in enumerate(LeadStatus.ALL):
            count = stage_counts.get(stage, 0)
            color = stage_colors.get(stage, "#ffffff")
            percentage = (count / total_leads * 100) if total_leads > 0 else 0
            
            with stage_cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{stage}</div>
                    <div class="metric-value" style="color: {color};">{count}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="background: {color}; width: {percentage}%;"></div>
                    </div>
                    <div style="text-align: center; margin-top: 8px; font-size: 12px; color: #93a0ad;">
                        {percentage:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Priority Leads Section
        st.markdown("### 🎯 Priority Leads (Top 8)")
        
        # Calculate priorities
        priority_list = []
        for _, row in df.iterrows():
            score, _, _, _, _, _, time_left = compute_priority_for_lead_row(row, weights)
            
            # SLA calculation
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            if isinstance(sla_entered, str):
                try: 
                    sla_entered = datetime.fromisoformat(sla_entered)
                except: 
                    sla_entered = datetime.utcnow()
            deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
            remaining = deadline - datetime.utcnow()
            overdue = remaining.total_seconds() <= 0
            
            # Predicted conversion
            prob = None
            if lead_model is not None:
                try: 
                    prob = predict_lead_probability(lead_model, row)
                except: 
                    prob = None
            
            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "No name",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": float(remaining.total_seconds() / 3600.0),
                "priority_score": score,
                "status": row.get("status"),
                "sla_overdue": overdue,
                "sla_deadline": deadline,
                "conversion_prob": prob,
                "damage_type": row.get("damage_type", "Unknown")
            })
        
        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)
        
        if not pr_df.empty:
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                status_color = stage_colors.get(r["status"], "#ffffff")
                
                # Priority badge color
                if score >= 0.7:
                    priority_color = "#ef4444"
                    priority_label = "🔴 CRITICAL"
                elif score >= 0.45:
                    priority_color = "#f97316"
                    priority_label = "🟠 HIGH"
                else:
                    priority_color = "#22c55e"
                    priority_label = "🟢 NORMAL"
                
                # SLA status
                if r["sla_overdue"]:
                    sla_html = f"<span style='color:#ef4444;font-weight:700;'>❗ OVERDUE</span>"
                else:
                    hours_left = int(r['time_left_hours'])
                    mins_left = int((r['time_left_hours'] * 60) % 60)
                    sla_html = f"<span style='color:#2563eb;font-weight:600;'>⏳ {hours_left}h {mins_left}m left</span>"
                
                # Conversion probability
                conv_html = ""
                if r["conversion_prob"] is not None:
                    conv_pct = r["conversion_prob"] * 100
                    conv_color = "#22c55e" if conv_pct > 70 else ("#f97316" if conv_pct > 40 else "#ef4444")
                    conv_html = f"<span style='color:{conv_color};font-weight:600;margin-left:12px;'>📊 {conv_pct:.0f}% Win Prob</span>"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="margin-bottom: 8px;">
                                <span style="color:{priority_color};font-weight:700;font-size:14px;">{priority_label}</span>
                                <span class="stage-badge" style="background:{status_color}20;color:{status_color};border:1px solid {status_color}40;">
                                    {r['status']}
                                </span>
                            </div>
                            <div style="font-size: 18px; font-weight: 700; color: #ffffff; margin-bottom: 4px;">
                                #{int(r['id'])} — {r['contact_name']}
                            </div>
                            <div style="font-size: 13px; color: #93a0ad; margin-bottom: 8px;">
                                {r['damage_type'].title()} | Est: <span style="color:#22c55e;font-weight:700;">${r['estimated_value']:,.0f}</span>
                            </div>
                            <div style="font-size: 13px;">
                                {sla_html}
                                {conv_html}
                            </div>
                        </div>
                        <div style="text-align: right; padding-left: 20px;">
                            <div style="font-size: 36px; font-weight: 700; color:{priority_color};">
                                {score:.2f}
                            </div>
                            <div style="font-size: 11px; color: #93a0ad; text-transform: uppercase;">
                                Priority
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No priority leads to display.")
        
        st.markdown("---")
        
        # Detailed Lead Cards (Expandable)
        st.markdown("### 📋 All Leads")
        
        for lead in leads:
            status_color = stage_colors.get(lead.status, "#ffffff")
            est_val = lead.estimated_value or 0
            
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'Unknown'} — ${est_val:,.0f}"
            
            with st.expander(card_title, expanded=False):
                # Lead info section
                colA, colB = st.columns([3, 1])
                
                with colA:
                    st.markdown(f"""
                    <div style="padding: 10px; background: rgba(255,255,255,0.02); border-radius: 8px;">
                        <div style="margin-bottom: 8px;">
                            <span style="color:#93a0ad;">Source:</span> 
                            <strong>{lead.source or '—'}</strong>
                            <span style="margin-left: 16px; color:#93a0ad;">Assigned:</span> 
                            <strong>{lead.assigned_to or '—'}</strong>
                        </div>
                        <div style="margin-bottom: 8px;">
                            <span style="color:#93a0ad;">Address:</span> 
                            <strong>{lead.property_address or '—'}</strong>
                        </div>
                        <div style="margin-bottom: 8px;">
                            <span style="color:#93a0ad;">Notes:</span> 
                            {lead.notes or '—'}
                        </div>
                        <div>
                            <span style="color:#93a0ad;">Created:</span> 
                            {lead.created_at.strftime('%Y-%m-%d %H:%M') if lead.created_at else '—'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with colB:
                    # SLA and status
                    entered = lead.sla_entered_at or lead.created_at
                    if isinstance(entered, str):
                        try: 
                            entered = datetime.fromisoformat(entered)
                        except: 
                            entered = datetime.utcnow()
                    
                    deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                    remaining = deadline - datetime.utcnow()
                    
                    if remaining.total_seconds() <= 0:
                        sla_status = f"<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>"
                    else:
                        hours = int(remaining.total_seconds() // 3600)
                        mins = int((remaining.total_seconds() % 3600) // 60)
                        sla_status = f"<div style='color:#2563eb;font-weight:600;'>⏳ {hours}h {mins}m</div>"
                    
                    st.markdown(f"""
                    <div style="text-align: right;">
                        <div class="stage-badge" style="background:{status_color}20;color:{status_color};border:1px solid {status_color}40;">
                            {lead.status}
                        </div>
                        <div style="margin-top: 12px;">
                            {sla_status}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Quick contact buttons
                qc1, qc2, qc3, qc4 = st.columns([1, 1, 1, 4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                
                if phone:
                    with qc1:
                        st.markdown(f"""
                        <a href='tel:{phone}' style='text-decoration:none;'>
                            <button style='background:#2563eb;color:#000;border:none;border-radius:8px;padding:8px 12px;cursor:pointer;width:100%;font-weight:600;'>
                                📞 Call
                            </button>
                        </a>
                        """, unsafe_allow_html=True)
                    
                    wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                    wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20following%20up%20on%20your%20restoration%20request."
                    with qc2:
                        st.markdown(f"""
                        <a href='{wa_link}' target='_blank' style='text-decoration:none;'>
                            <button style='background:#25D366;color:#000;border:none;border-radius:8px;padding:8px 12px;cursor:pointer;width:100%;font-weight:600;'>
                                💬 WhatsApp
                            </button>
                        </a>
                        """, unsafe_allow_html=True)
                
                if email:
                    with qc3:
                        st.markdown(f"""
                        <a href='mailto:{email}?subject=Follow%20up' style='text-decoration:none;'>
                            <button style='background:rgba(255,255,255,0.1);color:#fff;border:1px solid rgba(255,255,255,0.2);border-radius:8px;padding:8px 12px;cursor:pointer;width:100%;font-weight:600;'>
                                ✉️ Email
                            </button>
                        </a>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Lead update form
                with st.form(f"update_lead_{lead.id}"):
                    st.markdown("#### Update Lead")
                    
                    ucol1, ucol2 = st.columns(2)
                    with ucol1:
                        new_status = st.selectbox("Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status), key=f"status_{lead.id}")
                        new_assigned = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                        contacted = st.checkbox("Contacted", value=lead.contacted, key=f"contacted_{lead.id}")
                    
                    with ucol2:
                        inspection_scheduled = st.checkbox("Inspection Scheduled", value=lead.inspection_scheduled, key=f"insp_sched_{lead.id}")
                        inspection_completed = st.checkbox("Inspection Completed", value=lead.inspection_completed, key=f"insp_comp_{lead.id}")
                        estimate_submitted = st.checkbox("Estimate Submitted", value=lead.estimate_submitted, key=f"est_sub_{lead.id}")
                    
                    new_notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")
                    
                    if st.form_submit_button("💾 Update Lead"):
                        lead.status = new_status
                        lead.assigned_to = new_assigned
                        lead.contacted = contacted
                        lead.inspection_scheduled = inspection_scheduled
                        lead.inspection_completed = inspection_completed
                        lead.estimate_submitted = estimate_submitted
                        lead.notes = new_notes
                        
                        s.add(lead)
                        s.commit()
                        st.success(f"Lead #{lead.id} updated!")
                        st.rerun()
                
                # Estimates section
                st.markdown("#### 💰 Estimates")
                lead_estimates = s.query(Estimate).filter(Estimate.lead_id == lead.id).all()
                
                if lead_estimates:
                    for est in lead_estimates:
                        est_status = "✅ Approved" if est.approved else ("❌ Lost" if est.lost else "⏳ Pending")
                        est_color = "#22c55e" if est.approved else ("#ef4444" if est.lost else "#f97316")
                        
                        st.markdown(f"""
                        <div style="padding:10px;background:rgba(255,255,255,0.03);border-radius:8px;margin:8px 0;">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div>
                                    <span style="color:{est_color};font-weight:700;">{est_status}</span>
                                    <span style="margin-left:12px;color:#22c55e;font-weight:700;font-size:18px;">
                                        ${est.amount:,.0f}
                                    </span>
                                </div>
                                <div style="color:#93a0ad;font-size:12px;">
                                    {est.created_at.strftime('%Y-%m-%d') if est.created_at else '—'}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No estimates yet.")
                
                # Create estimate form
                with st.form(f"create_estimate_{lead.id}"):
                    st.markdown("**Create New Estimate**")
                    est_amount = st.number_input("Amount ($)", min_value=0.0, step=100.0, key=f"est_amt_{lead.id}")
                    est_details = st.text_area("Details", key=f"est_det_{lead.id}")
                    
                    if st.form_submit_button("➕ Create Estimate"):
                        create_estimate(s, lead.id, est_amount, est_details)
                        st.success("Estimate created!")
                        st.rerun()
