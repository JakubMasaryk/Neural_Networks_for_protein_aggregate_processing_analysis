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



-- ANN binary-classification training data (v3)
-- WT vs. ase1 and num1 mutants
drop procedure if exists p_ANN_binary_classification_v3;
delimiter //
create procedure p_ANN_binary_classification_v3(in p_starting_timepoint int, in p_ending_timepoint int)
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
		sacm.mutated_gene_standard_name in ('ASE1', 'NUM1')
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
			else 'disrupted_clearance'
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
		sacm.mutated_gene_standard_name in ('-', 'ASE1', 'NUM1') and
		sacm.metal_concentration > 0 and
		fnaa.number_of_foci > 0 and
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) >= p_starting_timepoint and
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) <= p_ending_timepoint;
end //
delimiter ;



-- ANN binary-classification training data (v4)
-- control vs. CHX-exposed cells
drop procedure if exists p_ANN_binary_classification_v4;
delimiter //
create procedure p_ANN_binary_classification_v4(in p_starting_timepoint int, in p_ending_timepoint int)
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
		saci.inhibitor_abbreviation= 'CHX'
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
		saci.inhibitor_solvent= 'DMSO' and
		saci.inhibitor_solvent_concentration= 1 and
		saci.inhibitor_abbreviation= '-' and
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
		'lower_aggregate_formation' as category
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
		saci.inhibitor_abbreviation= 'CHX' and
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) >= p_starting_timepoint and
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) <= p_ending_timepoint;
end //
delimiter ;



-- ANN multiclass-classification training data (v1)
-- WT vs. cin8', tpm1, tpm2, myo4 and she3 mutants sv. LatA-exposed cells
drop procedure if exists p_ANN_multi_class_classification_v1;
delimiter //
create procedure p_ANN_multi_class_classification_v1(in p_starting_timepoint int, in p_ending_timepoint int)
begin
	select -- complete dataset
		*
	from
	(
	select -- mutants with their corresponding controls
		*
	from
	(
	with  
	cte_selected_mutant_exprimnets as
	(
	select distinct
		sacm.date_label,
		e.microscopy_initial_delay_min,
		e.microscopy_interval_min
	from
		strains_and_conditions_main as sacm
	inner join
		experiments as e
	on
		e.date_label= sacm.date_label
	where
		sacm.mutated_gene_standard_name in ('CIN8', 'TPM1', 'TPM2', 'MYO4', 'SHE3') 
	),
	cte_control_mutant_data as
	(
	select
		fnaa.date_label,
		fnaa.experimental_well_label,
		-- sacm.mutation,
		-- fnaa.timepoint,
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) as timepoint_minutes,
		fnaa.fov_cell_id,
        fnaa.number_of_foci,
		fnaa.total_foci_area/fnaa.number_of_foci as single_focus_avg_area,
		'control' as category
	from
		strains_and_conditions_main as sacm
	inner join
		cte_selected_mutant_exprimnets as cte1
	on
		cte1.date_label= sacm.date_label
	inner join
		experimental_data_scd_foci_number_and_area as fnaa
	on
		sacm.date_label=fnaa.date_label and
		sacm.experimental_well_label= fnaa.experimental_well_label
	where
		sacm.mutated_gene_standard_name= '-' and
		sacm.metal_concentration= 0.5 and
		fnaa.number_of_foci > 0
	)
	select 
		*
	from
		cte_control_mutant_data
	union all
	select
		fnaa.date_label,
		fnaa.experimental_well_label,
		-- sacm.mutation,
		-- fnaa.timepoint,
		fnaa.timepoint * cte1.microscopy_interval_min - (cte1.microscopy_interval_min - cte1.microscopy_initial_delay_min) as timepoint_minutes,
		fnaa.fov_cell_id,
        fnaa.number_of_foci,
        fnaa.total_foci_area/fnaa.number_of_foci as single_focus_avg_area,
		'partially_disrupted_movement' as category
	from
		strains_and_conditions_main as sacm
	inner join
		cte_selected_mutant_exprimnets as cte1
	on
		cte1.date_label= sacm.date_label
	inner join
		experimental_data_scd_foci_number_and_area as fnaa
	on
		sacm.date_label=fnaa.date_label and
		sacm.experimental_well_label= fnaa.experimental_well_label
	where
		sacm.mutated_gene_standard_name in ('CIN8', 'TPM1', 'TPM2', 'MYO4', 'SHE3') and
		sacm.metal_concentration= 0.5 and
		fnaa.number_of_foci > 0
	) as selected_mutants_data
	union all
	select -- LatA-exposed with their corresponding controls
		*
	from
	(
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
		fnaa.number_of_foci > 0
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
		fnaa.number_of_foci > 0
	) as lata_exposed_data
	) as complete_dataset
where
	complete_dataset.timepoint_minutes >= p_starting_timepoint and
    complete_dataset.timepoint_minutes <= p_ending_timepoint;
end //
delimiter ;



-- experimentally verified mutants having a lower aggregate formation
drop procedure if exists p_lower_formation_mutants_scd_data;
delimiter //
create procedure p_lower_formation_mutants_scd_data(in p_starting_timepoint int, in p_ending_timepoint int)
	begin
	with
	cte_ts_experiments as
	(
	select
		e.date_label,
		e.microscopy_initial_delay_min,
		e.microscopy_interval_min
	from
		experiments as e
	inner join
		experiment_types as et
	on
		e.experiment_type_id= et.experiment_type_id
	where
		et.experiment_type= 'TS collection screening' and
		et.experiment_subtype= 'first round' and
		e.data_quality= 'Good'
	),
	cte_initital_cell_count_filter as
	(
	select
		sacm.date_label,
		sacm.experimental_well_label
		-- caac.timepoint,
		-- caac.number_of_cells
	from
		strains_and_conditions_main as sacm
	inner join
		cte_ts_experiments as cte1
	on
		sacm.date_label= cte1.date_label
	inner join
		experimental_data_sbw_cell_area_and_counts as caac
	on
		sacm.date_label= caac.date_label and
		sacm.experimental_well_label= caac.experimental_well_label
	where
		caac.timepoint= 1 and
		caac.number_of_cells > 100
	),
	cte_selected_experiments as
	(
	select distinct
		sacm.date_label,
		cte1.microscopy_initial_delay_min,
		cte1.microscopy_interval_min
	from
		strains_and_conditions_main as sacm
	inner join
		cte_ts_experiments as cte1
	on
		sacm.date_label=cte1.date_label
	inner join
		cte_initital_cell_count_filter as cte3
	on
		sacm.date_label= cte3.date_label and
		sacm.experimental_well_label= cte3.experimental_well_label
	where
		sacm.mutated_gene_standard_name in ('MTR3', 'NOG1', 'NOP4', 'NOP2')
	)
	select
		sacm.date_label,
		sacm.experimental_well_label,
		sacm.mutation,
		fnaa.timepoint * cte2.microscopy_interval_min - (cte2.microscopy_interval_min - cte2.microscopy_initial_delay_min) as timepoint_minutes,
		fnaa.fov_cell_id,
		fnaa.number_of_foci,
		ifnull(fnaa.total_foci_area/fnaa.number_of_foci, 0) as single_focus_avg_area
	from
		strains_and_conditions_main as sacm
	inner join
		cte_selected_experiments as cte2
	on
		sacm.date_label= cte2.date_label
	inner join
		experimental_data_scd_foci_number_and_area as fnaa
	on
		sacm.date_label= fnaa.date_label and
		sacm.experimental_well_label= fnaa.experimental_well_label
	where
		sacm.mutated_gene_standard_name in ('-', 'MTR3', 'NOG1', 'NOP4', 'NOP2') and
        fnaa.timepoint * cte2.microscopy_interval_min - (cte2.microscopy_interval_min - cte2.microscopy_initial_delay_min) >= p_starting_timepoint and
        fnaa.timepoint * cte2.microscopy_interval_min - (cte2.microscopy_interval_min - cte2.microscopy_initial_delay_min) <= p_ending_timepoint;
end //
delimiter ;
