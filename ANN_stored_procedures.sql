-- ANN binary-classification training data (v1)
-- control vs LatA-exposed cells
drop procedure if exists p_ANN_binary_classification_v1;
delimiter //
create procedure p_ANN_binary_classification_v1(in p_starting_timepoint int, in p_ending_timepoint int)
begin
	with
	cte_selected_inhibitor_experiments as
	(
	select distinct
		sacm.date_label,
		e.microscopy_initial_delay_min,
		e.microscopy_interval_min
	from
		strains_and_conditions_main as sacm
	inner join
		strains_and_conditions_inhibitor as saci
	on
		sacm.date_label= saci.date_label and
		sacm.experimental_well_label= saci.experimental_well_label
	inner join
		experiments as e
	on
		sacm.date_label= e.date_label
	where
		saci.inhibitor_abbreviation= 'LATA'
	),
	cte_control_inhibitor_data as
	(
	select
		sacm.date_label,
		sacm.experimental_well_label,
		-- saci.inhibitor_concentration as lata_concentration,
		-- saci.inhibitor_abbreviation,
		-- fnaa.timepoint,
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) as timepoint_minutes,
		fnaa.fov_cell_id,
		fnaa.number_of_foci,
		fnaa.total_foci_area/fnaa.number_of_foci as single_focus_avg_area,
		'control' as category
	from
		strains_and_conditions_main as sacm
	inner join
		cte_selected_inhibitor_experiments as cte1
	on
		sacm.date_label= cte1.date_label
	inner join
		strains_and_conditions_inhibitor as saci
	on
		sacm.date_label= saci.date_label and
		sacm.experimental_well_label= saci.experimental_well_label
	inner join
		experimental_data_scd_foci_number_and_area as fnaa
	on
		sacm.date_label=fnaa.date_label and
		sacm.experimental_well_label= fnaa.experimental_well_label
	where
		sacm.metal_concentration= 0.5 and
		(saci.inhibitor_solvent= 'DMSO' or  saci.inhibitor_solvent= '-') and
		(saci.inhibitor_solvent_concentration= 1 or saci.inhibitor_solvent_concentration= 0) and
		saci.inhibitor_abbreviation= '-' and
		fnaa.number_of_foci > 0 and
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) >= p_starting_timepoint and
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) <= p_ending_timepoint
	)
	select
		*
	from
		cte_control_inhibitor_data
	union all
	select distinct
		sacm.date_label,
		sacm.experimental_well_label,
		-- saci.inhibitor_concentration as lata_concentration,
		-- saci.inhibitor_abbreviation,
		-- fnaa.timepoint,
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) as timepoint_minutes,
		fnaa.fov_cell_id,
		fnaa.number_of_foci,
		fnaa.total_foci_area/fnaa.number_of_foci as single_focus_avg_area,
		'disrupted_movement' as category
	from
		strains_and_conditions_main as sacm
	inner join
		cte_selected_inhibitor_experiments as cte1
	on
		sacm.date_label= cte1.date_label
	inner join
		strains_and_conditions_inhibitor as saci
	on
		sacm.date_label= saci.date_label and
		sacm.experimental_well_label= saci.experimental_well_label
	inner join
		experimental_data_scd_foci_number_and_area as fnaa
	on
		sacm.date_label=fnaa.date_label and
		sacm.experimental_well_label= fnaa.experimental_well_label
	where
		sacm.metal_concentration= 0.5 and
		saci.inhibitor_abbreviation= 'LatA' and
        -- saci.inhibitor_concentration= 10 and
		fnaa.number_of_foci > 0 and
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) >= p_starting_timepoint and
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) <= p_ending_timepoint;
	end //
delimiter ;



-- ANN binary-classification training data (v2)
-- WT vs. tpm1, tmp2 and myo4 mutants
drop procedure if exists p_ANN_binary_classification_v2;
delimiter //
create procedure p_ANN_binary_classification_v2(in p_starting_timepoint int, in p_ending_timepoint int)
begin
	with
	cte_selected_experiments as
	(
	select distinct
		e.date_label,
		e.microscopy_initial_delay_min,
		e.microscopy_interval_min
	from
		experiments as e
	inner join
		strains_and_conditions_main as sacm
	on
		e.date_label=sacm.date_label
	where
		sacm.mutated_gene_standard_name in ('TPM1', 'TPM2', 'MYO4')
	)
	select
		sacm.date_label,
		sacm.experimental_well_label,
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) as timepoint_minutes,
		fnaa.fov_cell_id,
		fnaa.number_of_foci,
		fnaa.total_foci_area/fnaa.number_of_foci as single_focus_avg_area,
		case
			when sacm.mutated_gene_standard_name = '-' then 'control' 
			else 'disrupted_movement'
		end as category
	from
		strains_and_conditions_main as sacm
	inner join
		cte_selected_experiments as cte1
	on
		sacm.date_label= cte1.date_label
	inner join
		experimental_data_scd_foci_number_and_area as fnaa
	on
		sacm.date_label= fnaa.date_label and
		sacm.experimental_well_label= fnaa.experimental_well_label
	where
		sacm.mutated_gene_standard_name in ('-', 'TPM1', 'TPM2', 'MYO4') and
		sacm.metal_concentration > 0 and
		fnaa.number_of_foci > 0 and
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) >= p_starting_timepoint and
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) <= p_ending_timepoint;
end //
delimiter ;
